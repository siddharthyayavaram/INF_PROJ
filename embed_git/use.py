#Script for the Universal Sentence Encoder notebook file to run in HPC

import tensorflow as tf
import tensorflow_hub as hub

# Load the USE model
module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
use_model = hub.load(module_url)

import pickle

def save_arr(sentence_embeddings,name):
    with open(name, 'wb') as f:
        pickle.dump(sentence_embeddings, f)

def read_text(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for i in range(len(lines)):
        lines[i]=lines[i].strip()

    return lines

from sklearn.neighbors import NearestNeighbors

import numpy as np

def knn(e,sentence_embeddings,k=100):
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(sentence_embeddings)
    distances, indices = nbrs.kneighbors([e])
    indices=np.squeeze(indices)
    return indices

def s_score(i,i1,i2):
    common = np.intersect1d(np.intersect1d(i, i1), i2)
    s = len(common)

    print("Common elements:", common)
    print("Number of common elements:", s)
    return common

def indices(s,sentence_embeddings):
    print(s)
    e = np.squeeze(use_model([s]))
    i = knn(e,sentence_embeddings)
    # print(i)
    return i

sentences =read_text("generic.txt")
sentence_embeddings = use_model(sentences)

# sentence_embeddings = torch.tensor(sentence_embeddings.numpy())
# save_arr(sentence_embeddings,'bbc_e.pkl')

print(sentence_embeddings.shape)

s="its very easy"
i = indices(s,sentence_embeddings)
i1= indices("its very simple",sentence_embeddings)
i2= indices("its extremely easy",sentence_embeddings)

# for x in i:
#     print(sentences[x])
#     print()

# for x in i1:
#     print(sentences[x])
#     print()

# for x in i2:
#     print(sentences[x])
#     print()

y= s_score(i,i1,i2)

for x in y:
    print(sentences[x])
    print()