import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB


Xneg = np.load("Embedded_tweets/embeddings_neg_dim20.npy")
Xpos = np.load("Embedded_tweets/embeddings_pos_dim20.npy")
Ypos = np.ones(Xpos.shape[0])
Yneg = np.ones(Xneg.shape[0])*-1
X = np.concatenate((Xneg,Xpos))
Y = np.concatenate((Yneg, Ypos))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=5)
GNB = GaussianNB()
y_pred = GNB.fit(X_train, Y_train).predict(X_test)

print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (Y_test != y_pred).sum()))
# Y_test = []
# clf = svm.SVC()
# clf.fit(X_train,Y_train)
# Y_test = clf.predict(X_test)

# print("Number of mislabeled points out of a total %d points : %d"% (X_test.shape[0], (Y_test != y_pred).sum()))