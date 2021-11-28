from scipy.sparse import *
import numpy as np
import pickle

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

with open('cooc.pkl', 'rb') as f2:
    cooc = pickle.load(f2)

line = "vinco tresorpack 6 ( difficulty 10 of 10 object : disassemble and reassemble the wooden"
test = [vocab.get(t, -1) for t in line.strip().split()]
print(test)
test = [t for t in test if t >= 0]
print(test)

print(cooc.max())

np.random.seed(123)
xs = np.random.normal(size=(20, 4))
print(xs)
print(xs[0], xs[0, :])
nmax = 100
embedding_dim = 20
np.save("Embeddings/embeddings_nmax-" + str(nmax) + "_dim-" + str(embedding_dim), xs)