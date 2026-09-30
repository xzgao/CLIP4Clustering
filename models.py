import clip
import torch
from torch import nn
from timm.models.layers import trunc_normal_


class CLIPModel(nn.Module):
    def __init__(self, model_name="ViT-B/32"):
        super().__init__()
        self.clip, self.preprocess = clip.load(model_name, device="cuda")

    @property
    def dtype(self):
        return self.clip.visual.conv1.weight.dtype

    def encode_image(self, image):
        image_features = self.clip.visual(image.type(self.dtype))
        return image_features

    def encode_text(self, text):
        x = self.clip.token_embedding(text).type(self.dtype)

        x = x + self.clip.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip.text_projection

        return x



# ========== myCLIPModel：On the original CLIPModel, we added learnable prompts for the text ==========
class myCLIPModel(nn.Module):
    def __init__(self, model_name="ViT-B/32"):
        super().__init__()
        self.clip, self.preprocess = clip.load(model_name, device="cuda")
    @property
    def dtype(self):
        return self.clip.visual.conv1.weight.dtype

    def encode_image(self, image):
        image_features = self.clip.visual(image.type(self.dtype))
        return image_features

    def encode_text(self, text_tokens, ctx=None):
        """
        text_tokens: [B, 77]
        ctx: [n_ctx, width]
        """
        x = self.clip.token_embedding(text_tokens).type(self.dtype)

        if ctx is not None and ctx.shape[0] > 0:
            B, seq_len, width = x.shape
            n_ctx = ctx.shape[0]

            original_eot_pos = text_tokens.argmax(dim=-1)
            if (original_eot_pos + n_ctx >= seq_len).any():
                raise ValueError("Sentence too long for the current n_ctx")

            if n_ctx >= seq_len:
                raise ValueError(f"n_ctx={n_ctx} must be smaller than context length={seq_len}")

            ctx_cast = ctx.to(device=x.device, dtype=x.dtype)
            ctx_expand = ctx_cast.unsqueeze(0).expand(B, -1, -1)

            x = torch.cat(
                [
                    x[:, :1, :],                 # SOS
                    ctx_expand,                  # Insert ctx：1 ~ n_ctx
                    x[:, 1:seq_len - n_ctx, :],  
                ],
                dim=1,
            )
            eot_pos = text_tokens.argmax(dim=-1) + n_ctx
        else:
            eot_pos = text_tokens.argmax(dim=-1)

        x = x + self.clip.positional_embedding[:x.shape[1]].to(device=x.device, dtype=x.dtype)

        x = x.permute(1, 0, 2)
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip.ln_final(x).type(self.dtype)

        batch_indices = torch.arange(x.shape[0], device=x.device)

        x = x[batch_indices, eot_pos] @ self.clip.text_projection
        return x


class ClusterHead(nn.Module):
    def __init__(self, clip_model: myCLIPModel, in_dim=512, num_clusters=10, n_ctx=16, proj_dim=128):
        super().__init__()
        self.num_clusters = num_clusters
        self.n_ctx = n_ctx
        self.clip_model = clip_model
        # Freeze CLIP backbone
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # Learnable prompts
        ctx_dim = in_dim
        ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=torch.float32)
        nn.init.normal_(ctx_vectors, std=0.02)
        self.ctx = nn.Parameter(ctx_vectors)

        # ========= cluster‑level head（softmax） =========
        self.cluster_head_text = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_clusters),
            nn.Softmax(dim=1),
        )
        self.cluster_head_image = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, num_clusters),
            nn.Softmax(dim=1),
        )

        # ========= instance‑level head =========
        self.instance_head_text = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )
        
        self.instance_head_image = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, proj_dim),
        )

        trunc_normal_(self.cluster_head_text[0].weight, std=0.02)
        trunc_normal_(self.cluster_head_text[3].weight, std=0.02)
        trunc_normal_(self.cluster_head_image[0].weight, std=0.02)
        trunc_normal_(self.cluster_head_image[3].weight, std=0.02)

        trunc_normal_(self.instance_head_text[0].weight, std=0.02)
        trunc_normal_(self.instance_head_text[3].weight, std=0.02)
        trunc_normal_(self.instance_head_image[0].weight, std=0.02)
        trunc_normal_(self.instance_head_image[3].weight, std=0.02)

    def encode_with_ctx(self, text_tokens: torch.Tensor):
        """
        Input text token tensor [M,77]
        return: text_emb [M,512]
        """
       
        ctx_cast = self.ctx.to(self.clip_model.dtype)
        text_emb = self.clip_model.encode_text(text_tokens, ctx_cast)
        return text_emb

    # forward(image_feat, text_feat)：The first parameter is image features, the second parameter is text features! Don't mix them up!
    def forward(self, image_feat, text_feat):
        """
        Input the encoded image/text features [B,512]
        return:
            logit_text_cluster, logit_image_cluster: cluster‑level clustering assignment [B, num_clusters]
            feat_text_inst, feat_image_inst: instance‑level feature [B, proj_dim]
        """

        # cluster‑level output
        logit_text_cluster = self.cluster_head_text(text_feat)
        logit_image_cluster = self.cluster_head_image(image_feat)

        # instance‑level output
        feat_text_inst = self.instance_head_text(text_feat)
        feat_text_inst = nn.functional.normalize(feat_text_inst, dim=1)

        feat_image_inst = self.instance_head_image(image_feat)
        feat_image_inst = nn.functional.normalize(feat_image_inst, dim=1)

        return logit_text_cluster, logit_image_cluster, feat_text_inst, feat_image_inst
        

    def forward_embedding(self, image):
        embedding = self.cluster_head_image[0](image)
        embedding = self.cluster_head_image[1](embedding)
        embedding = self.cluster_head_image[2](embedding)
        embedding = self.cluster_head_image[3](embedding)
        return embedding

