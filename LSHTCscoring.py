import cPickle


def macrof1(ytest,ypred):
    tpset=dict();
    testset=dict();
    predset=dict();
    
    for i in range(len(ytest)):
        for predlabel in ypred[i]:
            if predlabel in predset:
                predset[predlabel]+=1;
            else:
                predset[predlabel]=1;
        for truelabel in ytest[i]:
            if truelabel in testset:
                testset[truelabel]+=1;
            else:
                testset[truelabel]=1;
            if truelabel in ypred[i]:
                if truelabel in tpset:
                    tpset[truelabel]+=1;
                else:
                    tpset[truelabel]=1;
    
    totalC=len(testset);
    
    MaP=0.0;
    MaR=0.0;
    
    if len(tpset)==0:
        return 0;
    
    for key in testset:
        if key in tpset:
            tp=float(tpset[key]);
            tpfp=float(predset[key]);
            tpfn=float(testset[key]);
            tmpP=float(tp/tpfp);
            tmpR=float(tp/tpfn);
        else:
            tmpP=0.0;
            tmpR=0.0;
        
        MaP+=tmpP;
        MaR+=tmpR;
    
    MaP=MaP/totalC;
    MaR=MaR/totalC;
    
    return 2*MaP*MaR/(MaP+MaR);
    
            
ycvpred=cPickle.load(open("./meta data/ycvpred.p",'rb'));
ycvtest=cPickle.load(open('./meta data/ycvtest.p','rb'));


print macrof1(ycvtest,ycvpred)

print macrof1([(1,2,3)],[(2,)])

print macrof1([(1,2,3),(1,2,3)],[(2,),(1,2)])


print macrof1([(1,2,3),(1,2,3)],[(2,),(1,)])

print macrof1([(1,2,3),(1,2,3)],[(2,),(1,2)])


print macrof1([(1,2,3),(1,2,3)],[(2,),(1,2,3)])

print macrof1([(1,2,3),(1,2,3)],[(2,),(1,2,3,4)])

print macrof1([(1,2,3),(1,2,3)],[(2,),(1,2,3,4,5)])

    
#test writing file

f = open('./submission/testouput.csv', 'w')
f.write("Id,Predicted\n");
import csv

cw=csv.writer(f);

for i,key in enumerate(ycvpred):
    nw=list();
    nw.append(i);
    for cp in key:
        nw.append(int(cp))
    cw.writerow(nw);

f.close();