from scipy.sparse import *
import numpy as np
import pickle

with open('vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)

with open('cooc.pkl', 'rb') as f2:
    cooc = pickle.load(f2)

line = "vinco tresorpack 6 ( difficulty 10 of 10 object : disassemble and reassemble the wooden"
test = [vocab.get(t, -1) for t in line.strip().split()]

test = [t for t in test if t >= 0]

a = np.array([None for _ in range(20)])
a = np.vstack((a, np.empty(20)))
print(a, a.shape)