import torch
import numpy as np
import os
import pandas as pd
import clip
import torch.nn.functional as F
from txt2excel import convert_cluster_txt_to_excel
import random
from models import myCLIPModel, ClusterHead
from eval_utils import cluster_metric
from torch.utils.data import DataLoader, TensorDataset
from loss_utils import DistillLoss, consistency_loss, entropy, InstanceDistillLoss
from retrieval_text import retrieve_text_for_batch



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True

def mine_nearest_neighbors_batch(feat:torch.Tensor, topk:int=50):
    B, D = feat.shape
    sim = feat @ feat.T
    mask = torch.eye(B, device=feat.device, dtype=torch.bool)
    sim.masked_fill_(mask, -1e9)
    _, indices = torch.topk(sim, k=topk, dim=-1)
    return indices


def infer(model, dataloader):
    model.eval()
    preds = []
    logits_image = []
    with torch.no_grad():
        for iter, (image,) in enumerate(dataloader):
            image = image.cuda()
            _, logit_image, _, _ = model(image, image)
            pred = torch.argmax(logit_image, dim=1).cpu().numpy()
            preds.append(pred)
            logits_image.append(logit_image.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    logits_image = np.concatenate(logits_image, axis=0)
    return preds, logits_image


if __name__ == "__main__":
    set_seed(42)
    dataset = "ImageNet-10"   # ["CIFAR-10", "CIFAR-20", "STL-10", "ImageNet-10", "ImageNet-Dogs", "DTD", "CIFAR-100"]
    my_n_ctx = 12
    topK = 50

    epochs = 200
    batch_size = 1024
    my_tau_1 = 0.5
    my_tau_2 = 1

    if dataset == "CIFAR-10" or dataset == "STL-10" or dataset == "ImageNet-10":
        cluster_num = 10
    elif dataset == "CIFAR-20":
        cluster_num = 20
    elif dataset == "ImageNet-Dogs":
        cluster_num = 15
    elif dataset == "DTD":
        cluster_num = 47
    elif dataset == "CIFAR-100":
        cluster_num = 100
    else:
        raise NotImplementedError

    images_embedding_train = np.load("./data/" + dataset + "_image_embedding_train.npy")
    images_embedding_train = images_embedding_train / np.linalg.norm(
        images_embedding_train, axis=1, keepdims=True
    )
    images_embedding_test = np.load("./data/" + dataset + "_image_embedding_test.npy")
    images_embedding_test = images_embedding_test / np.linalg.norm(
        images_embedding_test, axis=1, keepdims=True
    )
    labels_test = np.loadtxt("./data/" + dataset + "_labels_test.txt")

    csv_path = f"./data/{dataset}_selected_texts.csv"
    df = pd.read_csv(csv_path, header=None)
    candidate_sentences = df[0].tolist()
    text_tokens = clip.tokenize(candidate_sentences, truncate=True).cuda()

    my_clip_backbone = myCLIPModel("ViT-B/32").cuda()
    model = ClusterHead(clip_model=my_clip_backbone, in_dim=512, num_clusters = cluster_num, n_ctx=my_n_ctx).cuda()

    dataset_image_train = TensorDataset(torch.from_numpy(images_embedding_train).float())
    dataset_image_test = TensorDataset(torch.from_numpy(images_embedding_test).float())

    dataloader_train = DataLoader(
        dataset_image_train, batch_size=batch_size, shuffle=True, drop_last=True
    )
    dataloader_test = DataLoader(
        dataset_image_test, batch_size=batch_size, shuffle=False, drop_last=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), betas=(0.9, 0.99))
    distill_loss = DistillLoss(class_num=cluster_num, temperature=my_tau_1)
    instance_distill_loss = InstanceDistillLoss(temperature=my_tau_2)

    result_file_path = "./cluster_result.txt"
    with open(result_file_path, "w", encoding="utf-8") as f:
        f.write(f"dataset={dataset}, topK={topK}, temperature1={my_tau_1}, temperature2={my_tau_2}\n")
        f.write("epoch\tnmi\tacc\tari\n")

    print("Start training...")
    for epoch in range(epochs):
        model.train()
        loss_distill_epoch = loss_instance_distill_epoch = loss_consist_epoch = loss_entropy_epoch = 0
        for iter, (img_feat_batch,) in enumerate(dataloader_train):
            img_feat_batch = img_feat_batch.cuda()

            candidate_text_emb = model.encode_with_ctx(text_tokens)
            candidate_text_emb = candidate_text_emb.float()
            candidate_text_emb = F.normalize(candidate_text_emb, dim=1)

            txt_feat_batch = retrieve_text_for_batch(
                image_feat_batch=img_feat_batch,
                candidate_text_emb=candidate_text_emb,
                tau=0.005
            )

            batch_neigh_img_idx = mine_nearest_neighbors_batch(img_feat_batch, topk=topK)
            batch_neigh_txt_idx = mine_nearest_neighbors_batch(txt_feat_batch, topk=topK)

            B = img_feat_batch.size(0)
            select_k = torch.randint(0, topK, size=(B,), device=img_feat_batch.device)
            neigh_img_local_idx = batch_neigh_img_idx[torch.arange(B), select_k]
            neigh_txt_local_idx = batch_neigh_txt_idx[torch.arange(B), select_k]

            neigh_img_feat = img_feat_batch[neigh_img_local_idx]
            neigh_txt_feat = txt_feat_batch[neigh_txt_local_idx]

            logit_text, logit_image, feat_txt_inst, feat_img_inst = model(img_feat_batch, txt_feat_batch)
            neigh_logit_text, neigh_logit_image, neigh_feat_txt_inst, neigh_feat_img_inst = model(neigh_img_feat, neigh_txt_feat)

            loss_distill = distill_loss(logit_image, neigh_logit_text) + distill_loss(logit_text, neigh_logit_image)        # cluster_level
            loss_inst_distill = instance_distill_loss(feat_img_inst, neigh_feat_txt_inst) + instance_distill_loss(feat_txt_inst, neigh_feat_img_inst)  # instance_level
            loss_consist = consistency_loss(logit_text, logit_image)
            loss_entropy = entropy(logit_text) + entropy(logit_image)  # entropy-based regularization term

            #loss = loss_inst_distill + loss_distill + loss_consist - 5 * loss_entropy
            loss = loss_inst_distill + 2 * loss_distill + loss_consist - 2 * loss_entropy

            optimizer.zero_grad()
            loss.backward()

            # =========Debug: Check ctx gradients and updates=========
            if iter == 0:
                print(f"\n--- Debug ctx info ---")
                print(f"model.ctx.requires_grad = {model.ctx.requires_grad}")
                if model.ctx.grad is not None:
                    print(f"ctx grad norm: {model.ctx.grad.norm().item():.6f}")
                else:
                    print("⚠️ ctx.grad is None, no gradient is being backpropagated!")

                ctx_before_step = model.ctx.detach().clone()

            optimizer.step()

            loss_instance_distill_epoch += loss_inst_distill.item()
            loss_distill_epoch += loss_distill.item()
            loss_consist_epoch += loss_consist.item()
            loss_entropy_epoch += loss_entropy.item()

            if (iter + 1) % 50 == 0 or iter + 1 == len(dataloader_train):
                print(
                    "[Epoch {}/{}] [Iter {}/{}] InstanceLoss Distill: {:.4f} Loss Distill: {:.4f} Loss Consist: {:.4f} Loss Entropy: {:.4f}".format(
                        epoch + 1,
                        epochs,
                        iter + 1,
                        len(dataloader_train),
                        loss_inst_distill.item(),
                        loss_distill.item(),
                        loss_consist.item(),
                        loss_entropy.item(),
                    )
                )

        print(
            "[Epoch: {}] InstanceLoss Distill: {:.4f} Loss Distill: {:.4f} Loss Consist: {:.4f} Loss Entropy: {:.4f}".format(
                epoch + 1,
                loss_instance_distill_epoch / (iter + 1),
                loss_distill_epoch / (iter + 1),
                loss_consist_epoch / (iter + 1),
                loss_entropy_epoch / (iter + 1),
            )
        )

        preds, confidences_image = infer(model, dataloader_test)
        nmi, acc, ari = cluster_metric(labels_test, preds)

        # Append the results of this epoch to the txt
        with open(result_file_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch+1}\t{nmi:.4f}\t{acc:.4f}\t{ari:.4f}\n")
        # ----------------------------------------------------------------------

        # Call the packaged function to generate an Excel file
        convert_cluster_txt_to_excel(
            txt_file=result_file_path,
            excel_file="./cluster_result.xlsx"
        )
