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
def build_k_indices(y, k_fold, seed = 1):
    """build k indices for k-fold."""
    num_row = len(y)
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
def cooc_CV(k = 2, seed = 1, VOCAB_PATH = 'vocab.pkl',TWIT_POS_PATH = 'twitter-datasets/train_pos.txt', TWIT_NEG_PATH = 'twitter-datasets/train_neg.txt'):
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    
    counter = 1
    data, row, col = [], [], []
    coocList = []
    for fn in [TWIT_POS_PATH, TWIT_NEG_PATH]:
        with open(fn) as file:
            """SPlitting data into k parts"""
            f = file.readlines()
            splits = [range(len(f))]
            if(k != 1):
                """generates indieces twice very stupid pls fix"""
                splits = build_k_indices(f,k,seed)            
             
                
                    
            cooc_matrices = []
            for i in range(len(splits)):
                if len(cooc_matrices) < k:  
                    data.append([[],[]])
                    row.append([[],[]])
                    col.append([[],[]])
                 
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
                            """generates testing cooc"""
                            data[i][0].append(1)
                            row[i][0].append(t)
                            col[i][0].append(t2)
                            """Generates training cooc"""
                            for notK in range(k):
                                if notK  != k:
                                    data[i][1].append(1)
                                    row[i][1].append(t)
                                    col[i][1].append(t2)
                
                if counter % 10000 == 0:
                    print(counter)
                counter += 1
    for i in range(k):
        cooc = coo_matrix((data[i][0], (row[i][0], col[i][0])))
        """cooc2 is the one with most of the data"""
        cooc2 = coo_matrix((data[i][1], (row[i][1], col[i][1])))
        print("summing duplicates (this can take a while)")
        cooc.sum_duplicates()
        cooc2.sum_duplicates()
        totCooc = [cooc2, cooc]
        if len(cooc_matrices) < k:
            cooc_matrices.append(totCooc)
        else:
            cooc_matrices[i] 
                
    if(k == 1):
        with open('test.pkl', 'wb') as f:
            pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)
            return cooc
    else:
        name = 'Cooc_CV_seed%s_k%x.pkl'%(seed, k)
        with open(name, 'wb') as f:
            pickle.dump(cooc_matrices, f, pickle.HIGHEST_PROTOCOL)
        return cooc_matrices
def main():

    cooc_CV(2)

if __name__ == '__main__':
    main()
