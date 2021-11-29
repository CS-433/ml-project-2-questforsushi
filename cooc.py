#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle

"""
This function creates & store a coo matrix
The cooc matrix will contain (word1, word2) = n 
Where n indicates how many times word1 and word 2 appear together in the whole corpus
-> (word1, word1) = n will indicate word1 appears n times in the whole corpus 
if they appear k times in a same tweet, they are counted k times 
-> n could equal 0, if they never appear together 
Correspondance : the word corresponding to the word1 index can be found in cut_vocab[word1]"""

def main():
    with open('vocab_full.pkl', 'rb') as f:
        vocab = pickle.load(f)
    vocab_size = len(vocab)

    data, row, col = [], [], []
    counter = 1
    for fn in ['twitter-datasets/train_pos.txt', 'twitter-datasets/train_neg.txt']:
        with open(fn) as f:
            for line in f:
                # -1 is used instead of none for tokens not found in vocab
                tokens = [vocab.get(t, -1) for t in line.strip().split()]
                tokens = [t for t in tokens if t >= 0]
                for t in tokens:
                    for t2 in tokens:
                        # data is just used for construction of the matrix in the future (that's why
                        # the appended value is always 1)
                        data.append(1)
                        row.append(t)
                        col.append(t2)

                if counter % 10000 == 0:
                    print(counter)
                counter += 1
    cooc = coo_matrix((data, (row, col)))
    print("summing duplicates (this can take a while)")
    cooc.sum_duplicates()
    with open('cooc_full.pkl', 'wb') as f:
        pickle.dump(cooc, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
