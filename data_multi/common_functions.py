 #! python

import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import metrics

# finding the threshold based on probability and labe;s of the validation set    
def find_thr(pred,label):
    
    #find the best threshold where false possitive  rate and falsi negative points cross
    minn=100000
    thrr=0.4
    
    for thr in np.arange(0.1,1,0.1):
        prr = np.where(pred > thr, 1, 0)
        tn, fp, fn, tp = confusion_matrix(label,prr).ravel()
        if tp+fn > 0:
            frr=fn/(tp+fn)
        else:
            frr = 0
        if tn+fp > 0:    
            far=fp/(tn+fp)
        else:
            far = 0 
        if np.abs(frr - far) < minn:
            minn=np.abs(frr - far)
            thrr=thr
            
    return thrr

#subsample equally for imbalace dataset by selecting samples randomly
#x and y are the data and the corresponding labels
def balanced_subsample(x,y):
    class_xs = []
    min_elems = None
    subsample_size=1.0 # keep equal number of samples for all classes
    
    for yi in np.unique(y): # for any class in our data
        elems = x[(y == yi)] #find the data of each label
        class_xs.append((yi, elems)) 
        if min_elems == None or elems.shape[0] < min_elems: # find min_elemns as the group with lower samples
            min_elems = elems.shape[0]

    use_elems = min_elems # if proportion is not 1 not applicable in our work
    if subsample_size < 1:
        use_elems = int(min_elems*subsample_size)

    xs = []
    ys = []

    for ci,this_xs in class_xs: # randomly select samples from larger group
        if len(this_xs) > use_elems:
            np.random.shuffle(this_xs)

        x_ = this_xs[:use_elems]
        y_ = np.empty(use_elems)
        y_.fill(ci)
        
        xs.append(x_)
        ys.append(y_)
        
    # concatenate the data of all groups before returning them 
    xs = np.concatenate(xs)
    ys = np.concatenate(ys)
    
    return xs,ys

#calculate the performance (Accuracy, sensitivity, specificity, AUC)
def performance_calculation(array1,array2,array3):
     tn, fp, fn, tp = confusion_matrix(array1,array2).ravel()
     total=tn+fp+fn+tp
     acc= (tn+tp)/total
     sen = tp/(tp+fn)
     sps = tn/(tn+fp)
     
     fpr, tpr, thresholds = metrics.roc_curve(array1, array3)
     roc_auc=metrics.auc(fpr, tpr)
     
     return acc,sen,sps,roc_auc 
