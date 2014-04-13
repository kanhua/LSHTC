"""Calculate the simlarity """


import numpy as np;
import cPickle;


xtrain=cPickle.load(open("./meta data/trainSparseMtx.p",'rb'));
ytrain=cPickle.load(open('./meta data/y_train.p', 'rb'))

totalrow=xtrain.shape[0];

#setup a matrix for storing the relative simliarity
covMtx=np.zeros((xtrain.shape[0],xtrain.shape[0]));

#setup an array for string the norm of each row
rownorm=np.zeros((xtrain.shape[0],));

for i in range(totalrow):
    w=xtrain[i,:].todense();
    rownorm[i]=np.linalg.norm(w)
    
    
for i in range(totalrow):
    wi=xtrain[i,:].todense();
    for j in range(i+1, totalrow, 1):    
        wj=xtrain[j,:].todense();
        ip=np.inner(wi,wj)/(rownorm[i]*rownorm[j]);
        covMtx[i,j]=ip;
        covMtx[j,i]=ip;
    
np.save("./meta data/convMtx.npy",covMtx);
        
np.save("./meta data/trainNorm.npy",rownorm);