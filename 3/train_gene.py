'''
用所有的feature跑模型
'''
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# Samaneh Kouchaki
# This file includes ensemble lclassifiers including RF/ERF/Adaboost/bagging/GBT
# It also includes threshold setting

class EnsembleLearning:
    #filename1 is the feature matrix (rows:isolates and cols: features) and filename2 is labels one for each isolates
    def __init__(self, feat,label):
#         self.feat = pd.read_csv(filename1,header=None)
        self.feat=feat.values
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

        #data
        wdata = self.feat #features
        wlabel = self.label #labels
        
        print("====="+str(method)+"=======")
        for i in range(3):
            if method == "rf":
                
                #models depending which one you wwant to select:
                model = RandomForestClassifier(n_estimators=100, max_depth=None, max_features=1000 )
            elif method == "xgb":
                model = XGBClassifier(tree_method = 'gpu_hist')
            elif method == "ctb":
                model = CatBoostClassifier()
            elif method =="lr":
                model = LogisticRegression()
            else:
                print("model name error !!!!")
            #balancing the data before test and train
            data,label=self.balanced_subsample(wdata,wlabel)
            
            #5 fold CV
            cv=StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
            stratified_5folds = cv.split(data, label)
            
            for trind, teind in stratified_5folds:
                #80% of the data
                tr=wdata[trind]
                trl=wlabel[trind]
                
                # 20% of the data for final test
                te=wdata[teind]
                tel=wlabel[teind]
                
                train, test, train_label, test_label = train_test_split(tr,trl, test_size=0.2, random_state=2020)
                model.fit(train,train_label)

#                 print(model.predict_proba(test))
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

       
        print('acc:', np.mean(accm),np.std(accm))
        print('sen:', np.mean(senm),np.std(senm))
        print('sps:', np.mean(spsm),np.std(spsm))
        print('auc:', np.mean(roc_aucm),np.std(roc_aucm))
        
        f = open("/htju/gaocl/result_exp/3_data/res/gene/single/res.txt",'a+')
        f.write("====="+str(method)+"====="+str(drug)+"======")
        f.write('\tacc:'+str(np.mean(accm))+"\t"+str(np.std(accm)))
        f.write('\tsen:'+str(np.mean(senm))+"\t"+str(np.std(senm)))
        f.write('\tsps:'+str(np.mean(spsm))+"\t"+str(np.std(spsm)))
        f.write('\tauc:'+str(np.mean(roc_aucm))+"\t"+str(np.std(roc_aucm)))
        f.write("\n")
           
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
    
    #subsample equally for imbalace dataset by selecting samples randomly
    #x and y are the data and the corresponding labels
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



import numpy as np
import pandas as pd



drug_label = {'KANAMYCIN', 'CAPREOMYCIN', 'RIFAMPICIN', 'ETHIONAMIDE', 'ISONIAZID', 'OFLOXACIN', 'STREPTOMYCIN', 'ETHAMBUTOL', 'PROTHIONAMIDE', 'AMIKACIN', 'CIPROFLOXACIN', 'RIFABUTIN', 'PYRAZINAMIDE', 'MOXIFLOXACIN'}
# drug_label = ['MOXIFLOXACIN']
# CYCLOSERINE 'PARA-AMINOSALICYLIC ACID'  LEVOFLOXACIN 
for drug in drug_label:

    # dataset_columns = np.load("/htju/gaocl/result_exp/2_data/nu/single_feature2/"+drug+"_nuheader.npy", allow_pickle=True)
    # # dataset_columns = np.append(dataset_columns,np.array(['label']))
    # print("****************************************")
    # #直接加载转变好的dataset
    print(str(drug)+"   loader dataset ...")
    
    # # dataset = pd.read_csv("./dataset_pff/data3/"+str(drug)+".csv")
    # dd = np.load("/htju/gaocl/result_exp/2_data/nu/single_feature2/"+drug+"_nu.npy")
    # dataset = pd.DataFrame(dd, columns=dataset_columns)
    dataset = pd.read_csv("/htju/gaocl/result_exp/2_data/gene/single_feature/"+drug+".csv")
    X_data = dataset[dataset.columns[0:-1]]
    # X_data = X_data.loc[:,~((X_data==0).all())]
    print(X_data.shape)
    label = dataset[dataset.columns[-1]]
    print(label.shape)

    # if len(set(label.values))<2:
    #     print(drug, "=label just 1 class ==pass==")
    #     continue
    del dataset
    print(set(label.values))
    x=EnsembleLearning(X_data,label)
#     # print('Ensemble Learning: F1 for INH')
#     # print('******** F1 + RF')
    x.ensemble_methods('rf', str(drug))
    x.ensemble_methods('xgb', str(drug))
    x.ensemble_methods('ctb', str(drug))
    x.ensemble_methods('lr', str(drug))
