import networkx as nx
import matplotlib.pyplot as plt;
import csv
import numpy as np
G=nx.DiGraph()


with open('./source data/hierarchy.txt', 'rb') as f:
    reader = csv.reader(f,delimiter=' ')
    for row in reader:
        edge=tuple([int(row[0]),int(row[1])]);
        G.add_edge(*edge)


#nx.draw(G,with_labels=False)

#plt.show()

minp=100000;
minNode=0;
countleave=0;
leavenodes=list();
for d in G.nodes_iter(data=False):
    parentd=G.predecessors(d)
    parentnum=len(parentd);
    if parentnum<minp:
        minp=parentnum;
        minNode=d;
    childd=G.successors(d)
    if len(childd)==0:
        countleave=countleave+1;
        leavenodes.append(d)
        
print parentnum,minNode;
print countleave;

np.save("./meta data/leavenodes.npy", np.array(leavenodes));
