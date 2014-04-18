"""This script calculates and output the norm of train and test matrix"""

from LSHTCUtil import *
import numpy as np
import scipy.sparse as sp;


trainMtx=np.load("./meta data/trainSparseMtx.npy").item();

testMtx=np.load('./meta data/testSparseMtx.npy').item();

print "calculating norm"
 
print "w"
wnorm=trainMtx.multiply(trainMtx);
print "v"      
vnorm=testMtx.multiply(testMtx);
wnorm=np.sqrt(wnorm.sum(axis=1));
vnorm=np.sqrt(vnorm.sum(axis=1));
wnorm=np.asarray(wnorm).squeeze();
vnorm=np.asarray(vnorm).squeeze();
np.savetxt("./meta data/trainmtx_norm.csv", wnorm, delimiter=',')
np.savetxt("./meta data/testmtx_norm.csv", vnorm, delimiter=',')
