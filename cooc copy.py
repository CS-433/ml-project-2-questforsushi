#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
from scipy.sparse import coo
from sklearn.model_selection import KFold

"""
This function creates & store a coo matrix
The cooc matrix will contain (word1, word2) = n 
Where n indicates how many times word1 and word 2 appear together in the whole corpus
-> (word1, word1) = n will indicate word1 appears n times in the whole corpus 
if they appear k times in a same tweet, they are counted k times 
-> n could equal 0, if they never appear together 
Correspondance : the word corresponding to the word1 index can be found in cut_vocab[word1]"""
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = len(y)
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
def cooc(k = 1):
    with open('vocab_full.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    
    counter = 1
    data, row, col = [], [], []
    coocList = []
    for fn in ['twitter-datasets/train_pos.txt', 'twitter-datasets/train_neg.txt']:
        with open(fn) as file:
            """SPlitting data into k parts"""
            f = file.readlines()
            splits = [range(len(f))]
            if(k != 1):
                """generates indieces twice very stupid pls fix"""
                splits = build_k_indices(f,k,1)            
             
                
                    
            cooc_matrices = []
            for i in range(len(splits)):
                if len(cooc_matrices) < k:  
                    data.append([])
                    row.append([])
                    col.append([])
                 
                part = splits[i]
                print(part)
                for index in part:
                    line = f[index]
                    # -1 is used instead of none for tokens not found in vocab
                    tokens = [vocab.get(t, -1) for t in line.strip().split()]
                    tokens = [t for t in tokens if t >= 0]
                    for t in tokens:
                        for t2 in tokens:
                            # data is just used for construction of the matrix in the future (that's why
                            # the appended value is always 1)
                            data[i].append(1)
                            row[i].append(t)
                            col[i].append(t2)
            
                if counter % 10000 == 0:
                    print(counter)
                counter += 1
    for i in range(k):
        cooc = coo_matrix((data[i], (row[i], col[i])))
        print("summing duplicates (this can take a while)")
        cooc.sum_duplicates()
        if len(cooc_matrices) < k:
            cooc_matrices.append(cooc)
        else:
            cooc_matrices[i] 
                
    if(k == 1):
        with open('test.pkl', 'wb') as f:
            pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)
            return cooc
    else:
        with open('tests.pkl', 'wb') as f:
            pickle.dump(cooc_matrices, f, pickle.HIGHEST_PROTOCOL)
        return cooc_matrices
def main():
    cooc(2)

if __name__ == '__main__':
    main()
