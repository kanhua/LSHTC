import numpy as np
from sklearn.cross_validation import KFold

import cPickle;


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
       
        dTrainMtx=trainMtx.todense();
        dTestMtx=testMtx[:,0:maxveclength].todense();
        wnorm=np.linalg.norm(dTrainMtx,axis=1);
        vnorm=np.linalg.norm(dTestMtx,axis=1);
        
        dTestMtx=np.transpose(dTestMtx);
        prd=dTrainMtx*dTestMtx;
        
        for i in range(prd.shape[0]):
            for j in range(prd.shape[1]):
                prd[i,j]=prd[i,j]/(wnorm[i]*vnorm[j])
        
        scorevec=np.transpose(np.asarray(prd));
    return scorevec;
        
        
xtrain=np.load("./meta data/trainSparseMtx.npy");
xtrain=xtrain.item();
ytrain=cPickle.load(open('./meta data/y_train.p', 'rb'))

kf = KFold(len(ytrain), n_folds=5, indices=True,shuffle=True)

for trainidx, testidx in kf:
    xcvtrain=xtrain[trainidx,:];
    ycvtrain=[ytrain[i] for i in trainidx];
    
    xcvtest=xtrain[testidx,:];
    ycvtest=[ytrain[i] for i in testidx];
    
    
sv=simscore(xcvtrain,xcvtest,sparseMode=False)

np.save("./meta data/simscore.npy",sv)
ycvpred=list();
for i in range(xcvtest.shape[0]):
    a=np.argsort(sv[i,:])[::-1];
    ycvpred.append(ycvtrain[a[0]]);
    
cPickle.dump(ycvtest,open('./meta data/ycvtest.p','wb'));
cPickle.dump(ycvpred,open("./meta data/ycvpred.p",'wb'));

print "done"

