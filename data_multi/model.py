
import pandas as pd
import numpy as np
from sklearn.model_selection import PredefinedSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from common_functions import *
from cvcodes import proba_mass_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier,XGBRFClassifier
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

warnings.filterwarnings('ignore') 
alexander = -1
# Samaneh Kouchaki
# This file includes multi and single label RF classifiers
#In all classifiers we run 10 runs of 5-fold crossvalidation in each run 20 percent keeps as test set and 80 percent train and validation sets
#for setting the classifier threshold (it is based on best accuracy on validation set)
#10 fold cross validation and 5 runs

class EnsembleLearning_multilabel:
    #load data and label
    #filename1 is the feature matrix (rows:isolates and cols: features) and filename2 is labels one for each isolates
    def __init__(self, feat_df,label_df):
        self.feat=feat_df.values
        self.label=label_df.values 
        # self.params = open("./params.txt","a+")
    #multi-label RF
    # name1: all mutations list and name2: important features list for all drugs 
    #name1 and name2 are for loading important features. It is used only if not none otherwise the whole feature space is used
    def ensemble_multilabel(self,name1 = None , name2 = None):
        #keeping performance
        # 10 runs of 5 fold (50) for EMB, INH, RIF, PZA, XDR, MDR (6)
        accm=np.zeros((50,4))
        senm=np.zeros((50,4))
        spsm=np.zeros((50,4))
        aucm=np.zeros((50,4))
        kk = 0
        
        wdata = self.feat
        wlabel = self.label
        wlabel = wlabel.astype(int)
        
        inds =[]
        
        
        for i in range(10):
            # random_parm ={
            #         'estimator__learning_rate':[0.01,0.05,0.1],
            #         'estimator__n_estimators': [50,100,150],
            #         'estimator__max_depth': range(5,15,3),  # 树的最大深度
            #         'estimator__min_child_weight':range(1,6,2),  # 决定最小叶子节点样本权重和，如果一个叶子节点的样本权重和小于min_child_weight则拆分过程结束。
            #         'estimator__gamma': [0, 0.05,0.1,0.2],  # 指定了节点分裂所需的最小损失函数下降值。这个参数的值越大，算法越保守
            #         'estimator__subsample':[0.7, 0.8,0.9],  # 每个决策树所用的子样本占总样本的比例（作用于样本）
            #         'estimator__colsample_bytree':[0.7, 0.8,0.9],
            #         'estimator__eta': [0.1,0.2,0.3,0.4,0.5],
            # }
        #  model = OneVsRestClassifier(XGBClassifier(tree_method = 'gpu_hist',gpu_id=1, ))
        #   model = RandomizedSearchCV(estimator = cf, param_distributions = random_parm, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs=4, scoring='f1_micro')

        #    model = OneVsRestClassifier(XGBRFClassifier(tree_method = 'gpu_hist',gpu_id=1, ))
            model = RandomForestClassifier(n_estimators=50, max_depth=None )
            test_fold = np.zeros(len(wlabel))  
            #use iterative split
        #   ind = iterative_split(wlabel, 5)
            ind = proba_mass_split(wlabel, 5)
            
            #split data based on iterative method
            for j in range(len(ind)):
                for k in range(len(ind[j])):
                    test_fold[ind[j][k]]=j
            ps = PredefinedSplit(test_fold)


            for trind, teind in ps.split():
                # 80% as train and validation
                train=wdata[trind]
                train_label=wlabel[trind]
       
                #20% as test
                test=wdata[teind]
                test_label=wlabel[teind]
             
                # fit the model 
                model.fit(train,train_label)
                #model = xgb.train(params, xgb.DMatrix(train, train_label))
              
                # find treshold for each label
                pred=model.predict(test)
                #accuracy per label
                for j in range(len(pred[0])):
                    acc,sen,sps,auc=performance_calculation(test_label[:,j],pred[:,j],pred[:,j])
                    accm[kk][j]=acc
                    senm[kk][j]=sen
                    spsm[kk][j]=sps
                    aucm[kk][j]=auc
                kk=kk+1
                print("--kk--", kk)
            #   break
        # print(model.best_params_)
        # self.params.write("best_params_: "+str(i)+ str(model.best_params_))
        # self.params.write("\n")
        # return 
        print("accm: ", accm*100, "\nsenm: ", senm*100, "\nspsm*100: ",spsm*100, "\naucm:", aucm*100)
        f = open("./rf.txt", "a+")
        f.write(str(name1))
        f.write("rf_res ={'accm': np.array("+str((accm*100).tolist())+"),\n'senm':np.array(" + str((senm*100).tolist())+"),\n'spsm': np.array("+str((spsm*100).tolist())+ "),\n'aucm' : np.array("+str((aucm*100).tolist())+")}\n")
        f.close()
                                        # return 
        return accm *100, senm*100, spsm*100, aucm*100
        

    def ensemble_multilabel2(self,name1 = None , name2 = None):
        accm=np.zeros((50,4))
        senm=np.zeros((50,4))
        spsm=np.zeros((50,4))
        aucm=np.zeros((50,4))
        kk = 0;
        
        wdata = self.feat
        wlabel = self.label
        wlabel = wlabel.astype(int)
        
      
        for i in range(5):
            
            model = OneVsRestClassifier(XGBClassifier(tree_method = 'gpu_hist',gpu_id=1, ))

            #5 fold CV
            kfold = KFold(n_splits=5, shuffle=True)
            index = kfold.split(X=wdata)
            
            for trind, teind in index:
                # 80% as train and validation
                tr=wdata[trind]
                trl=wlabel[trind]
                
                # 20% as test data
                te=wdata[teind]
                tel=wlabel[teind]
                
                  # train validation split based on iterative CV
                train, test, train_label, test_label = train_test_split(tr,trl, test_size=0.2, random_state=0)
                model.fit(train,train_label)
 
                 # find treshold for each label
                pred=model.predict_proba(test)
                # print(pred)
                thr=[]
                for j in range(len(pred[0])):
                    thr.append(find_thr(pred[:,j],test_label[:,j]))
                pred=model.predict_proba(te)
                prr=np.zeros((len(tel),len(tel[0])))
                
                for j in range(len(pred[0])):
                    prr[:,j]= np.where(pred[:,j] > thr[j], 1, 0)
                
                #accuracy per label
                for j in range(len(prr[0])):
                    acc,sen,sps,auc=performance_calculation(tel[:,j],prr[:,j],prr[:,j])
                    accm[kk][j]=acc
                    senm[kk][j]=sen
                    spsm[kk][j]=sps
                    aucm[kk][j]=auc
                kk=kk+1
                print("kk:",kk)
        f = open("./xgb_top_threshold.txt", "a+")
        f.write(str(name1))
        f.write("xgb_res ={'accm': np.array("+str((accm*100).tolist())+"),\n'senm':np.array(" + str((senm*100).tolist())+"),\n'spsm': np.array("+str((spsm*100).tolist())+ "),\n'aucm' : np.array("+str((aucm*100).tolist())+")}\n")
        f.close()
        # return 
        return accm*100, senm*100, spsm*100, aucm*100
