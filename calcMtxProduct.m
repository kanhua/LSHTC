% Calculate the product of sparse matrix
clear all;
xtrain=load('./meta data/trainSparseMtx.mat');
xtest=load('./meta data/testSparseMtx.mat');

xtrain=xtrain.xtrain;
xtest=xtest.xtrain;

xtrain=xtrain(1:end,:);
xtest=xtest(1:end,:);

sizediff=size(xtest,2)-size(xtrain,2);

if sizediff<0
    xtest=[xtest,spalloc(size(xtest,1),-sizediff,10)];
elseif sizediff>0
    xtrain=[xtrain,spalloc(size(xtrain,1),sizediff,10)];
end

mtxproduct=xtrain*transpose(xtest);