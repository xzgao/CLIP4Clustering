import clip
import torch
import numpy as np
import pandas as pd
from models import CLIPModel


nouns = pd.read_csv("./text/GPT_generated_texts_ImageNet_10.csv", header=None).values 
nouns_num = nouns.shape[0]
batch_size = 2048

model = CLIPModel(model_name="ViT-B/32").cuda()
model.eval()

features = []
print("Inferring text features for full‑sentence inputs")
for i in range(nouns_num // batch_size + 1):
    start = i * batch_size
    end = start + batch_size
    if end > nouns_num:
        end = nouns_num
    sentences_batch = nouns[start:end, 0].tolist()  # list[str]

    with torch.no_grad():
        prompt_tokens = clip.tokenize(sentences_batch, truncate=True).cuda()
        feature = model.encode_text(prompt_tokens)
        features.append(feature.cpu().numpy())

    if i % 50 == 0:
        print(f"[Completed {min(i * batch_size, nouns_num)}/{nouns_num}]")

features = np.concatenate(features, axis=0)
print("Feature shape:", features.shape)
np.save("./data/sentences_embedding.npy", features)
