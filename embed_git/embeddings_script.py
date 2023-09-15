#converted embeddings notebook to a python script to run in the HPC

import numpy as np
import random
import time
import pickle
from sentence_transformers import SentenceTransformer
sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')

import nltk
from nltk.corpus import wordnet
# nltk.download('wordnet')

import spacy
nlp = spacy.load("en_core_web_lg")  # Load the medium English model

from sklearn.neighbors import NearestNeighbors

def cosine(u, v):
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

def embed_1(s):
    return sbert_model.encode(s)

def embed(s):
    lst = s.split()
    l = len(lst)
    if(l<3):
        return sbert_model.encode(s)
    
    sentence = np.zeros(768)
    
    for i in range(0, l - 2):
        sub_sentence = " ".join(lst[i:i+3])
        sub_embedding = sbert_model.encode(sub_sentence)
        sentence += sub_embedding
    
    divisor = l - 2
    result_vector = sentence / divisor
    return result_vector

def make_embeddings(sentences):
    sentence_embeddings=[]
    for s in sentences[:]:
        sentence_embeddings.append(embed_1(s))
    return sentence_embeddings

def find_closest_synonyms(word, num_synonyms=10):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    synonyms = list(set(synonyms))
    
    num_synonyms = min(num_synonyms, len(synonyms))

    selected_synonyms = synonyms[:num_synonyms]
    
    return selected_synonyms

def remove_underscores(synonyms):
    cleaned_synonyms = [syn.replace('_', ' ') for syn in synonyms]
    return cleaned_synonyms

def find_most_sim(word,synonyms):
    maxsim=0
    sim=0
    fsyn=""
    for syn in synonyms:
        if ((word.lower() not in syn.lower()) and (syn.lower() not in word.lower()) and (len(word)>0 and len(syn)>0)):
            sim=nlp(word).similarity(nlp(syn))
            if sim>maxsim:
                maxsim=sim
                fsyn=syn

    return fsyn,maxsim

def select_random(sentence):
    doc = nlp(sentence)
    words = [token.text for token in doc if token.pos_ in ['VERB']]
    if(len(words)==0):
        return doc[random.randrange(len(sentence.split()))].text
    random_word_index = random.randrange(len(words))  # Generate a random index
    random_word = words[random_word_index]
    return random_word

def replace_w_aug(sentence):
    word=select_random(sentence)
    final,simval = find_most_sim(word,remove_underscores(find_closest_synonyms(word)))
    l=sentence.split()
    l[l.index(word)]=final
    l=' '.join(l)
    return l

def select_random_2(sentence):
    doc = nlp(sentence)
    words = [token.text for token in doc if token.pos_ in ['NOUN','ADJ']]
    if(len(words)==0):
        return doc[random.randrange(len(sentence.split()))].text
    random_word_index = random.randrange(len(words))  # Generate a random index
    random_word = words[random_word_index]
    return random_word

def replace_w_aug_2(sentence):
    word=select_random_2(sentence)
    final,simval = find_most_sim(word,remove_underscores(find_closest_synonyms(word)))
    l=sentence.split()
    l[l.index(word)]=final
    l=' '.join(l)
    return l

def knn(e,sentence_embeddings,k=10):
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
    e = embed_1(s)
    i = knn(e,sentence_embeddings)
    # print(i)
    return i

def read_text(file_path):

    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    for i in range(len(lines)):
        lines[i]=lines[i].strip()

    return lines

def save_arr(sentence_embeddings,name):
    with open(name, 'wb') as f:
        pickle.dump(sentence_embeddings, f)

def main():
    with open('bbc.pkl','rb') as f:
        sentences=pickle.load(f)

    # sentences =read_text("1_combined.txt")

    # sentence_embeddings = make_embeddings(sentences)
    # save_arr(sentence_embeddings,'bbc_e.pkl')

    with open('array_data_1.pkl', 'rb') as f:
        load = pickle.load(f)

    s="i couldnt go outside because its raining cats and dogs"
    i = indices(s,load)
    i1= indices(replace_w_aug(s),load)
    i2= indices(replace_w_aug_2(s),load)

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

if __name__=="__main__":
    start = time.time()
    main()
    end = time.time()
    print("The time of execution of above program is :",
        (end-start) * 10**3, "ms")