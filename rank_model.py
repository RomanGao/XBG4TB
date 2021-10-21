'''
用top importance feature 跑模型
'''
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
# import statsmodels.api as sm
from scipy import stats



# Samaneh Kouchaki
# This file includes ensemble lclassifiers including RF/ERF/Adaboost/bagging/GBT
# It also includes threshold setting

class EnsembleLearning:
    #filename1 is the feature matrix (rows:isolates and cols: features) and filename2 is labels one for each isolates
    def __init__(self, feat,label):
#         self.feat = pd.read_csv(filename1,header=None)
        self.feat=feat
#         self.label = pd.read_csv(filename2,header=None)
        self.label=label.values 
        self.label=label.ravel()

    #in all classifiers we run 10 times in each run 20 percent keeps as test set and 80 percent train 
    #from train set 20 percent considers as validation set and remaining for training. The validation set is used
    #for setting the classifier threshold (it is based on classification accuracy)
    #10 fold cross validation and 5 runs
    def ensemble_methods(self,method, drug):
        # for keeping performance (sensitivity, specificity, AUC, accuracy)
        accm=[]
        senm=[]
        spsm=[]
        roc_aucm=[]
        feat_imp = pd.read_csv("./dataset_pff/xgb/"+str(drug)+"_imp.csv")
        thold = [0.005, 0.001, 0.01 ]

        for th in thold:
            muts = feat_imp[feat_imp['importance']>=th].name.values.tolist()
            #data
            wdata = self.feat[muts].values #features
            wlabel = self.label #labels
            
            print("=="+str(th)+"==="+str(method)+"=======")
            for i in range(5):
                if method == "rf":
                    #models depending which one you wwant to select:
                    model = RandomForestClassifier(n_estimators=100, max_depth=None, max_features=1000 )
                elif method == "xgb":
                    model = XGBClassifier()
                elif method == "ctb":
                    model = CatBoostClassifier()
                else:
                    print("model name error !!!!")
                    break;
                #balancing the data before test and train
                # data,label=self.balanced_subsample(wdata,wlabel)
                data, label = wdata, wlabel
                model.fit(data,label)

                cv=StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
                stratified_5folds = cv.split(data, label)
                
                for trind, teind in stratified_5folds:
                    #80% of the data
                    tr=wdata[trind]
                    trl=wlabel[trind]
                    
                    # 20% of the data for final test
                    te=wdata[teind]
                    tel=wlabel[teind]
                    
                    model.fit(tr,trl)

                    
                    train, test, train_label, test_label = train_test_split(tr,trl, test_size=0.2, random_state=2020)
                    model.fit(train,train_label)
                    #print(model.predict_proba(test))
                    #find threshold using validation set
                    pred=model.predict_proba(test)[:,1]
                    thr=self.find_thr(pred,test_label)
                    # thr = 0.499
                    
                    # use threshold for test data
                    pred=model.predict_proba(te)[:,1]
                    prr = np.where(pred > thr, 1, 0)
                    
                    acc,sen,sps,roc_auc=self.performance_calculation(tel,prr,pred)
                    accm.append(acc)
                    senm.append(sen)
                    spsm.append(sps)
                    roc_aucm.append(roc_auc)

            print("\tmax\tmean\tstd")
            print('acc:', np.max(accm), np.mean(accm),np.std(accm))
            print('sen:', np.max(sen), np.mean(senm),np.std(senm))
            print('sps:', np.max(sps), np.mean(spsm),np.std(spsm))
            print('auc:', np.max(roc_aucm), np.mean(roc_aucm),np.std(roc_aucm))
            
            f = open("./dataset_pff/xgb/topPerform21_"+str(th)+".txt", "a+")
            f.write("====="+ str(method)+"===="+str(th)+"==="+str(drug)+"======")
            f.write('acc: max= ' +str(np.max(accm))+"\tmean= "+str(np.mean(accm))+"\t "+str(np.std(accm)))
            f.write('sen: max= ' +str(np.max(accm))+"\tmean= "+str(np.mean(senm))+"\t "+str(np.std(senm)))
            f.write('sps: max= ' +str(np.max(accm))+"\tmean= "+str(np.mean(spsm))+"\t "+str(np.std(spsm)))
            f.write('auc: max= ' +str(np.max(accm))+"\tmean= "+str(np.mean(roc_aucm))+"\t "+str(np.std(roc_aucm)))
            f.write("\n")
    def balanced_subsample(self,x,y):
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
  
     # finding the threshold based on probability and labels of the validation set    
    def find_thr(self,pred,label):
        
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
    
    #calculate the performance (Accuracy, sensitivity, specificity, AUC)  
    def performance_calculation(self,array1,array2,array3):
        # print("array1: ",array1)
        # print("array2: ",array2)
        tn, fp, fn, tp = confusion_matrix(array1,array2).ravel()
        # print("tn:", tn,"  fp:", fp," fn:",fn, " tp:",tp)
        # print()
        total=tn+fp+fn+tp
        acc= (tn+tp)/total #accuaracy
        sen = tp/(tp+fn) #sensitivity
        sps = tn/(tn+fp) # specificity

        fpr, tpr, thresholds = metrics.roc_curve(array1, array3)
        roc_auc=metrics.auc(fpr, tpr)

        return acc,sen,sps,roc_auc

import numpy as np
import pandas as pd

dataset_columns = np.load("./dataset_pff/header.npy", allow_pickle=True)
dataset_columns = np.append(dataset_columns,np.array(['label']))

drug_label = {'KANAMYCIN', 'CAPREOMYCIN', 'RIFAMPICIN', 'ETHIONAMIDE', 'ISONIAZID', 'OFLOXACIN', 'STREPTOMYCIN', 'ETHAMBUTOL', 'PROTHIONAMIDE', 'AMIKACIN', 'CIPROFLOXACIN', 'RIFABUTIN', 'PYRAZINAMIDE', 'MOXIFLOXACIN'}
# drug_label = {'CAPREOMYCIN', 'AMIKACIN', 'KANAMYCIN', 'RIFABUTIN', 'CIPROFLOXACIN'}
# AMIKACIN  KANAMYCIN RIFABUTIN CIPROFLOXACIN
for drug in drug_label:
    print("****************************************")
    #直接加载转变好的dataset
    print(str(drug)+"   loader dataset ...")
    
    # dataset = pd.read_csv("./dataset_pff/data3/"+str(drug)+".csv")
    dd = np.load("./dataset_pff/data3/"+str(drug)+".npy")
    dataset = pd.DataFrame(dd, columns=dataset_columns)
    
    X_data = dataset[dataset.columns[1:-1]]
    X_data = X_data.loc[:,~((X_data==0).all())]
    print(X_data.shape)
    label = dataset[dataset.columns[-1]]
    print(label.shape)

    if len(set(label.values))<2:
        print(drug, "=label just 1 class ==pass==")
        continue
    del dataset
    print(set(label.values))
    x=EnsembleLearning(X_data,label)
#     # print('Ensemble Learning: F1 for INH')
#     # print('******** F1 + RF')
    # x.ensemble_methods('rf', str(drug))
    x.ensemble_methods('xgb', str(drug))
    # x.ensemble_methods('ctb', str(drug))