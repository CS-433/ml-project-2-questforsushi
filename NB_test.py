import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification


Xneg = np.load("Embedded_tweets/tweet_embeddings_neg_dim_20.npy")
Xpos = np.load("Embedded_tweets/tweet_embeddings_pos_dim_20.npy")
Ypos = np.ones(Xpos.shape[0])
Yneg = np.ones(Xneg.shape[0])*-1
X = np.concatenate((Xneg, Xpos))
Y = np.concatenate((Yneg, Ypos))

# TODO : choose a good size for training the model, or try a GPU implementation
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.999, random_state=5)
# GNB = GaussianNB()
# y_pred = GNB.fit(X_train, Y_train).predict(X_test)

# print("Number of mislabeled points out of a total %d points NB : %d"% (X_test.shape[0], (Y_test != y_pred).sum()))
# y_pred = []
# clf = LogisticRegression(random_state=0).fit(X_train, Y_train)
# y_pred = clf.predict(X_test)

# print("Number of mislabeled points out of a total %d points logistic : %d"% (X_test.shape[0], (Y_test != y_pred).sum()))
# Y_test = []
print(Y_train.shape)
clf = make_pipeline(StandardScaler(), svm.LinearSVC(random_state=5, tol=1e-5, max_iter=10000))
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
#print((1 != y_pred or y_pred != -1).sum())

shape = X_test.shape[0]
missed = (Y_test != y_pred).sum()
accuracy = 1-missed/shape
print("Number of mislabeled points out of a total {} points : {}, giving an accuracy of {}".format(shape, missed, accuracy))