import numpy as np
from sklearn.cross_validation import KFold

import cPickle;


def simscore(trainMtx,testMtx):
    scorevec=np.zeros((testMtx.shape[0],trainMtx.shape[0]));

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
            ip=np.inner(w,v)/(wnorm*vnorm);
            scorevec[i,j]=ip;
    return scorevec;
        

xtrain=cPickle.load(open("./meta data/trainSparseMtx-min.p",'rb'));
ytrain=cPickle.load(open('./meta data/y_train-min.p', 'rb'))

kf = KFold(len(ytrain), n_folds=5, indices=True,shuffle=True)

for trainidx, testidx in kf:
    xcvtrain=xtrain[trainidx,:];
    ycvtrain=[ytrain[i] for i in trainidx];
    
    xcvtest=xtrain[testidx,:];
    ycvtest=[ytrain[i] for i in testidx];
    
    
sv=simscore(xcvtrain,xcvtest)
ycvpred=list();
for i in range(xcvtest.shape[0]):
    a=np.argsort(sv[i,:])[::-1];
    ycvpred.append(ycvtrain[a[0]]);
    
cPickle.dump(ycvtest,open('./meta data/ycvtest.p','wb'));
cPickle.dump(ycvpred,open("./meta data/ycvpred.p",'wb'));


