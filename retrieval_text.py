import torch
import numpy as np
import torch.nn.functional as F


def retrieve_text_for_batch(image_feat_batch: torch.Tensor, candidate_text_emb: torch.Tensor, tau=0.005):
    similarity = torch.matmul(image_feat_batch, candidate_text_emb.T)
    similarity = torch.softmax(similarity / tau, dim=1)
    retrieval_emb = similarity @ candidate_text_emb
    retrieval_emb = F.normalize(retrieval_emb, dim=1)
    return retrieval_emb
