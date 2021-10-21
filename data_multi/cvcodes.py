#! python
import itertools
import numpy as np
from math import ceil
import random
#Samaneh Kouchaki
# it contains two ways of multi-output stratification

#(1) multi output stratified cross validation based on distribution 
# y: labels
#folds: number of cross validation folds
def proba_mass_split(y, folds = 5):
    obs, classes = y.shape
    dist = y.sum(axis=0).astype('float')
    dist /= dist.sum()
    index_list = []
    fold_dist = np.zeros((folds, classes), dtype='float')
    for _ in range(folds):
        index_list.append([])
    for i in range(obs):
        if i < folds:
            target_fold = i
        else:
            x = fold_dist.sum(axis=1)
            x[x==0] = 0.001
            normed_folds = fold_dist.T / x 
            how_off = normed_folds.T - dist
            target_fold = np.argmin(np.dot((y[i] - .5).reshape(1, -1), how_off.T))
        fold_dist[target_fold] += y[i]
        index_list[target_fold].append(i)
    return index_list


#(2) multi output iterative stratified cross validation 
# it is based on spliting rare labels first
#(1) multi output stratified cross validation based on distribution 
# y: labels
#folds: number of cross validation folds
def iterative_split(y, folds = 5):
    index_list = []
    for _ in range(folds):
        index_list.append([])
    #build the dictionary for all possible option of labels combinations
    alld = []
    salld = []
    ind=list(range(0,len(y[0])))
    for i in range(len(y[0]),0,-1):   #for all first line drugs
        mdr = itertools.combinations(ind, i) #all possible combinations of length 4,3,2,1
        k = 0
        for k1 in mdr:
            alld.append(k1)
            salld.append(0)
    
    #calculate mdr statistics at least one positive sample
    for i in range(len(y)):
        kk = -1
        for j in alld:
            kk = kk+1
            flag = 0
            for k in j:
                if y[i][k] == 1:
                    flag = flag+1
            if flag == len(j):
                salld[kk] = salld[kk]+1
                break
    ss=0;        
    for i in range(len(y)): #all negative
        flag = 0 
        for k in range(len(y[0])):
            if y[i][k] == 0:
                flag = flag+1
            if flag == len(y[0]):
                ss = ss + 1
                break
    salld.append(ss) 
    alld.append((0,0,0,0,0,0))
    
    cj = []
    r = 1.0/folds
    for i in range(folds): #number of examples per subsets
        cj.append(ceil(len(y)*r))
        
    cij=np.zeros((len(salld),folds)) #desired number for each label option and fold
    for i in range(len(salld)):
        for j in range(folds):
            cij[i][j]=ceil(salld[i]*r)
            
    yy=list(range(0,len(y)))
    while len(y)>0:
        minv=20000;
        minind=-1
        for i in range(len(salld)):
            if salld[i] != 0 and salld[i] <=minv:
                minv = salld[i]
                minind = i
        rind=[]
        for i in range(len(y)):
            flag = 1;
            k=alld[minind]
            if sum(k) == 0 and len(k) == len(y[i][:]) and sum(y[i][:]) == 0:
                flag = 1
            elif sum(y[i][:]) != len(k):
                flag = 0
            else:
                for j in range(len(k)):
                    if y[i][k[j]] == 0:                    
                        flag = 0
            if flag == 1: #sample contains the value we are searching for
               maxind = [ii for ii in range(len(cij[minind][:])) if cij[minind][ii] == max(cij[minind][:])]
               if len(maxind) == 1:
                   m = maxind [0]
               else:
                   maxind1 = [ii for ii in range(len(cj)) if cj[ii] == max (cj)]
                   if len(maxind1) == 1:
                       m = maxind1 [0]
                   else:
                       m = random.choice(maxind1)
               index_list[m].append(yy[i])
               rind.append(i)
               cj[m] = cj[m] -1 
               cij[:][m] = cij[:][m] - 1
        y = np.delete(y,rind,0)
        yy = np.delete(yy,rind,0)
        salld[minind] = 0
    return index_list
