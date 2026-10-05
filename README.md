# CLIP4Clustering: A Novel Language-driven Pre-trained Contrastive Clustering Network

This is the code for the paper "CLIP4Clustering: A Novel Language-driven Pre-trained Contrastive Clustering Network" (submited to IEEE TETCI). Our key idea is to improve image clustering by leveraging the external textual semantics from the pre-trained model, in the absence of class name priors.



# Dependency

- pytorch>=2.0.1
- torchvision>=0.15.2
- munkres>=1.1.4
- scikit-learn>=1.2.2
- clip>=1.0
- timm>=0.9.2
- faiss-gpu>=1.7.4

# Usage

To improve the readability and extendibility of the code, we split different steps of our CLIP4Clustering method into separate `.py` files. Below is the step-by-step tutorial. Note that the intermediate results would be saved to the `./data` folder.

## Image and Text Embedding Inference (Take ImageNet-10 as an example)
We first need to compute the image embedding with the CLIP model by running

> python image_embedding.py

and we get
```text
./data/
├── ImageNet-10_image_embedding_train.npy
├── ImageNet-10_image_embedding_test.npy
├── ImageNet-10_labels_train.txt
├── ImageNet-10_labels_test.txt        
```

and the embedding of generated sentences (provided in the `./text` folder) for text space construction by running

> python text_embedding.py

and we get
```text
./data/
├── sentences_embedding.npy
```

## Candidate Text Construction
Next, we aim to find discriminative generated sentences to describe image semantic centers. (Here, we haven't introduced learnable prompts yet.)

1st running `filter_sentences.py`: Saving intermediate kmeans results, suggests rerun

> python filter_sentences.py

and we get
```text
./data/
├── ImageNet-10_image_43cluster
├── ...      
```

2nd running `filter_sentences.py`: Perform screening, generate `ImageNet-10_selected_texts` and `ImageNet-10_filtered_sentences_text.txt`

> python filter_sentences.py

and we get
```text
./data/
├── ImageNet-10_filtered_sentences_embedding.npy
├── ImageNet-10_filtered_sentences_text.txt
├── ImageNet-10_selected_texts
```


## Learnable Prompts and Cluster Heads Training
For better collaboration between image and text features, we train additional cluster heads and learnable prompts to further improve the clustering performance by running

> python train_head.py




# Dataset

CIFAR-10, CIFAR-20 (CIFAR-100), STL-10 will be automatically downloaded by Pytorch. ImageNet-10 and ImageNet-dogs are subsets of the ImageNet dataset, with class indices 
provided [here](https://github.com/Yunfan-Li/Contrastive-Clustering/tree/main/datasets). DTD could be downloaded from <https://www.robots.ox.ac.uk/~vgg/data/dtd/.> 

# Citation

If you find TAC useful in your research, please consider citing:
```
@article{Xiao2026CLIP4Clustering,
  title={CLIP4Clustering: A Novel Language-driven Pre-trained Contrastive Clustering Network},
  author={xxx},
  journal={xxxx},
  year={2026}
}
```
