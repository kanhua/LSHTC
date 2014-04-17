"""Convert the original data to npy files"""

from sklearn.datasets import load_svmlight_file
#from sklearn.naive_bayes import MultinomialNB #GaussianNB

#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.svm import LinearSVC
from numpy import *
from scipy.io import savemat

X_train, y_train = load_svmlight_file("./source data/test-sklearn.csv", multilabel=True);
X_train=X_train.astype(uint32);


#save("./meta data/testSparseMtx.npy",X_train);
savemat("./meta data/testSparseMtx.mat", {'xtrain':X_train});

import cPickle;
cPickle.dump(y_train, open('./meta data/y_test.p', 'wb')) ;
#cPickle.dump(X_train,open("./meta data/trainSparseMtx-min.p",'wb'));
#obj = cPickle.load(open('./meta data/y_train.p', 'rb'));
