import numpy as np
from sklearn.cross_validation import KFold

import cPickle;

def mtxproduct(trainMtx,testMtx):
      maxveclength=np.min([testMtx.shape[1],trainMtx.shape[1]]);
      dTrainMtx=trainMtx.todense().astype(float);
      dTestMtx=testMtx[:,0:maxveclength].todense().astype(float);
      wnorm=np.linalg.norm(dTrainMtx,axis=1);
      vnorm=np.linalg.norm(dTestMtx,axis=1);
    
      dTestMtx=np.transpose(dTestMtx);
      prd=dTrainMtx*dTestMtx;
      del dTrainMtx;
      del dTestMtx;
           
      for i in range(prd.shape[0]):
            for j in range(prd.shape[1]):
                  prd[i,j]=prd[i,j]/(wnorm[i]*vnorm[j])
      
      return prd;

def sparsemtxproduct(trainMtx,testMtx):
      trainMtx=trainMtx.astype(float);
      testMtx=testMtx.astype(float);
      
      print "transposing the matrix"
      dtrainMtx=trainMtx.transpose();      
      dtestMtx=testMtx.transpose();
      
      maxveclength=np.min([testMtx.shape[1],trainMtx.shape[1]]);
      
      print "calculating norm"
      
      print "w"
      wnorm=trainMtx.multiply(trainMtx);
      print "v"      
      vnorm=testMtx.multiply(testMtx);
      wnorm=np.sqrt(wnorm.sum(axis=1));
      vnorm=np.sqrt(vnorm.sum(axis=1));
      wnorm=np.asarray(wnorm).squeeze();
      vnorm=np.asarray(vnorm).squeeze();
      
      
      print "matrix multiplication"
      prd=trainMtx.dot(dtestMtx);
      prd=prd.todense();
      

      
      print "normalisation...."
      for i in range(prd.shape[0]):
            for j in range(prd.shape[1]):
                  prd[i,j]=prd[i,j]/(wnorm[i]*vnorm[j]) 
      return prd;

def getindex(i,split,bitesize,remem):
      if i<split:
            startidx=i*bitesize;
            endidx=(i+1)*bitesize;
      elif i==split:
            startidx=i*bitesize;
            endidx=startidx+remem;
      return startidx,endidx
      

def simscore(trainMtx,testMtx,sparseMode=True,split=2,split2=2):
      
      if sparseMode==True:    
            scorevec=np.zeros((testMtx.shape[0],trainMtx.shape[0]));
            maxveclength=np.min([testMtx.shape[1],trainMtx.shape[1]]);
          #setup an array for string the norm of each row
            rownorm=np.zeros((trainMtx.shape[0],));        
            for i in range(trainMtx.shape[0]):
                w=trainMtx[i,:];
                rownorm[i]=np.sqrt(w.dot(w.T))[0,0];
            
            for i in range(testMtx.shape[0]):
                w=testMtx[i,:];
                wnorm=np.sqrt(w.dot(w.T))[0,0];
                print "calculating",i;
                for j in range(trainMtx.shape[0]):
                    v=trainMtx[j,:];
                    vnorm=np.sqrt(v.dot(v.T))[0,0]
                    ip=w[0,0:maxveclength].multiply(v[0,0:maxveclength])/(wnorm*vnorm);
                    scorevec[i,j]=ip[0,0];
           
      else:
          #setup an array for storing the results       
          scorevec=np.zeros((testMtx.shape[0],trainMtx.shape[0]));  
             
          if split==0:
                prd=sparsemtxproduct(trainMtx, testMtx);
                scorevec=np.transpose(np.asarray(prd));
          else:
          
                #calcualte the split
                bitesize=np.floor_divide(trainMtx.shape[0],split);
                remem=np.mod(trainMtx.shape[0],split);
                
                bitesize2=np.floor_divide(testMtx.shape[0],split2);
                remem2=np.mod(testMtx.shape[0],split2);
                
                #setup an array for string the norm of each row
                
                
                for i in range(split+1):
                      for j in range(split2+1):
                        
                        startidx,endidx=getindex(i,split,bitesize,remem);
                        startidx2,endidx2=getindex(j,split2,bitesize2,remem2);
                              
                        prd=mtxproduct(trainMtx[startidx:endidx,:], testMtx[startidx2:endidx2,:])
                        scorevec[startidx2:endidx2,startidx:endidx]=np.transpose(np.asarray(prd));
                        del prd
                         
                                    
      return scorevec;
        
        
xtrain=np.load("./meta data/trainSparseMtx.npy");
xtrain=xtrain.item();

#for i in range(xtrain.shape[0]):
    #print "norm",i
    #a=xtrain[i,:].multiply(xtrain[i,:]);
    #a.sum();


ytrain=cPickle.load(open('./meta data/y_train.p', 'rb'))

kf = KFold(len(ytrain), n_folds=5, indices=True,shuffle=True)

for trainidx, testidx in kf:
    xcvtrain=xtrain[trainidx,:];
    ycvtrain=[ytrain[i] for i in trainidx];
    
    xcvtest=xtrain[testidx,:];
    ycvtest=[ytrain[i] for i in testidx];
    
    
sv=simscore(xcvtrain,xcvtest,sparseMode=False,split=0)
#sv2=simscore(xcvtrain,xcvtest,sparseMode=False,split=2)

np.save("./meta data/simscore.npy",sv)
ycvpred=list();
for i in range(xcvtest.shape[0]):
    a=np.argsort(sv[i,:])[::-1];
    ycvpred.append(ycvtrain[a[0]]);
    
cPickle.dump(ycvtest,open('./meta data/ycvtest.p','wb'));
cPickle.dump(ycvpred,open("./meta data/ycvpred.p",'wb'));

print "done"


