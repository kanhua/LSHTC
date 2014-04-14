
from LSHTCUtil import *
import numpy as np

import cPickle;

from sklearn.datasets import load_svmlight_file


def simscore(trainMtx,testMtx,sparseMode=True):
    
    if sparseMode==True:    
        scorevec=np.zeros((testMtx.shape[0],trainMtx.shape[0]));
        maxveclength=np.min([testMtx.shape[1],trainMtx.shape[1]]);
        #setup an array for string the norm of each row
        rownorm=np.zeros((trainMtx.shape[0],));        
        for i in range(trainMtx.shape[0]):
            w=trainMtx[i,:].todense();
            rownorm[i]=np.linalg.norm(w);
        
        for i in range(testMtx.shape[0]):
            w=testMtx[i,:].todense();
            wnorm=np.linalg.norm(w);
            print "calculating",i;
            for j in range(trainMtx.shape[0]):
                v=trainMtx[j,:].todense();
                vnorm=np.linalg.norm(v);
                ip=np.inner(w[0,0:maxveclength],v[0,0:maxveclength])/(wnorm*vnorm);
                scorevec[i,j]=ip;
       
    else:
       
        maxveclength=np.min([testMtx.shape[1],trainMtx.shape[1]]);
        #setup an array for string the norm of each row
        rownorm=np.zeros((trainMtx.shape[0],));
        dTrainMtx=trainMtx.todense();
        dTestMtx=testMtx[:,0:maxveclength].todense();
        wnorm=np.linalg.norm(dTrainMtx,axis=1);
        vnorm=np.linalg.norm(dTestMtx,axis=1);
        wv=wnorm*vnorm;
        prd=np.multiply(dTrainMtx,dTestMtx);
        innerp=prd.sum(axis=1);
        scorevec=innerp/wv;
        
    return scorevec;
        


xtrain=cPickle.load(open("./meta data/trainSparseMtx-min.p",'rb'));
ytrain=cPickle.load(open('./meta data/y_train-min.p', 'rb'))

xtest, ytest= load_svmlight_file("./source data/test-sk-min.csv", multilabel=True);

sv=simscore(xtrain,xtest,sparseMode=False);
ycvpred=list();

for i in range(xtest.shape[0]):
    a=np.argsort(sv[i,:])[::-1];
    ycvpred.append(ytrain[a[0]]);


ouputfilename='./submission/testouput.csv'
writePredict(ouputfilename, ycvpred);