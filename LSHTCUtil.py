import csv


def writePredict(filename,ypred):
    """Write prediction into the file for Kaggle"""
    f = open(filename, 'w')
    f.write("Id,Predicted\n");
    
    cw=csv.writer(f);
    
    for i,key in enumerate(ypred):
        nw=list();
        nw.append(i+1);
        for cp in key:
            nw.append(int(cp))
        cw.writerow(nw);
    
    f.close();    