#!/usr/bin/env python3
from scipy.sparse import *
import numpy as np
import pickle
import random


def glove(embedding_dim=20, alpha=3 / 4):
    print("loading cooccurrence matrix")
    with open('cooc_full.pkl', 'rb') as f:
        cooc = pickle.load(f)
    print("{} nonzero entries".format(cooc.nnz))

    nmax = 100  # this will serve as a cut-off to do not exagerate the importance of some very popular words
    print("using nmax =", nmax, ", cooc.max() =", cooc.max())

    print("initializing embeddings")
    xs = np.random.normal(size=(cooc.shape[0], embedding_dim))
    ys = np.random.normal(size=(cooc.shape[1], embedding_dim))

    eta = 0.001  # gradient descent

    epochs = 10

    for epoch in range(epochs):
        print("epoch {}".format(epoch))
        for ix, jy, n in zip(cooc.row, cooc.col, cooc.data):
            # we use the logarithm of the co-occurrence values
            log_co = np.log(n)
            fn = min(1, pow(n / nmax, alpha))
            x, y = xs[ix], ys[jy]
            grad_common_part = 2 * fn * (log_co - np.dot(x, y))
            xs[ix] += eta * grad_common_part * y
            ys[jy] += eta * grad_common_part * x

    np.save("Embeddings/embeddings_nmax" + str(nmax) + "_dim" + str(embedding_dim), xs)
    return xs


def main():
    glove()


if __name__ == '__main__':
    main()
