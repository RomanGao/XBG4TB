from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.metrics import classification_report

from sklearn.feature_selection import VarianceThreshold #方差过滤的方法
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2  #卡方检验
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('Agg')

def performance_calculation(array1,array2,array3):
    # print("array1: ",array1)
    # print("array2: ",array2)
    tn, fp, fn, tp = confusion_matrix(array1,array2).ravel()
    # print("tn:", tn,"  fp:", fp," fn:",fn, " tp:",tp)
    # print()
    total=tn+fp+fn+tp
    acc= (tn+tp)/total #accuaracy
    sen = tp/(tp+fn) #sensitivity
    sps = tn/(tn+fp) # specificity

    fpr, tpr, thresholds = roc_curve(array1, array3)
    roc_auc=auc(fpr, tpr)

    return acc,sen,sps,roc_auc

def evalate(wdata, wlabel,model,name):
    # for keeping performance (sensitivity, specificity, AUC, accuracy)
    accm=[]
    senm=[]
    spsm=[]
    roc_aucm=[]
    f = open("./data_eval/"+str(name)+"_res.txt", "a+")
    for i in range(3):
        cv=StratifiedKFold(n_splits=5, random_state=2020, shuffle=True)
        stratified_5folds = cv.split(wdata, wlabel)
        
        for trind, teind in stratified_5folds:
            #80% of the data
            tr=wdata[trind]
            trl=wlabel[trind]
            
            # 20% of the data for final test
            te=wdata[teind]
            tel=wlabel[teind]
            
            train, test, train_label, test_label = train_test_split(tr,trl, test_size=0.2, random_state=2020)
            model.fit(train,train_label)

            pred=model.predict_proba(test)[:,1]
            # thr=self.find_thr(pred,test_label)
            thr = 0.5
            
            # use threshold for test data
            pred=model.predict_proba(te)[:,1]
            prr = np.where(pred >= thr, 1, 0)
            
            acc,sen,sps,roc_auc=performance_calculation(tel,prr,pred)
            accm.append(acc)
            senm.append(sen)
            spsm.append(sps)
            roc_aucm.append(roc_auc)

            # classification_report(tel, prr, target_names=["0",'1'])
    print('acc:', np.mean(accm),np.std(accm))
    print('sen:', np.mean(senm),np.std(senm))
    print('sps:', np.mean(spsm),np.std(spsm))
    print('auc:', np.mean(roc_aucm),np.std(roc_aucm))

    f.write("====="+str(model)+"====="+str(drug)+"======")
    f.write('\tacc:'+str(np.mean(accm))+"\t"+str(np.std(accm)))
    f.write('\tsen:'+str(np.mean(senm))+"\t"+str(np.std(senm)))
    f.write('\tsps:'+str(np.mean(spsm))+"\t"+str(np.std(spsm)))
    f.write('\tauc:'+str(np.mean(roc_aucm))+"\t"+str(np.std(roc_aucm)))
    f.write("\n")
    
    return [accm, senm, spsm, roc_aucm]

def Filter(X,y):

    X_fsvar = VarianceThreshold(np.median(X.var().values)).fit_transform(X)
    # 等同于 X_var0 = selector.fit_transform(VarianceThreshold())
    score = []
    for i in range(3,200,5):
        X_fschi = SelectKBest(chi2,k=i).fit_transform(X_fsvar,y)
        acc,sen,sps,roc_auc = evalate(X_fschi,y,XGBClassifier(tree_method='gpu_hist', gpu_id=1),'xgb_filter')
        score.append(roc_auc)

    plt.plot(for i in range(3,200,5):,score)
    plt.legend()
    plt.savefig("./data_eval/xgb_filter.png")

    plt.close()

def Filter_p(X,y):
    X_fsvar = VarianceThreshold().fit_transform(X)
    chivalue, pvalues_chi = chi2(X_fsvar,y)
    #k取多少？我们想要消除所有p值大于设定值，比如0.05或0.01的特征：
    k = chivalue.shape[0] - (pvalues_chi > 0.05).sum()
    X_fschi = SelectKBest(chi2,k=k).fit_transform(X_fsvar,y)
    acc,sen,sps,roc_auc = evalate(X_fschi,y,XGBClassifier(tree_method='gpu_hist', gpu_id=1),'xgb_filter_p')

def Embedded(X,y):
    from sklearn.feature_selection import SelectFromModel

    threshold = np.linspace(0,(XGBClassifier(tree_method='gpu_hist', gpu_id=1).fit(X,y).feature_importances_).max(),20)
    # 等同于 X_var0 = selector.fit_transform(VarianceThreshold())
    score = []
    for i in range(3,200,5):
        X_fschi =SelectFromModel(XGBClassifier(tree_method='gpu_hist', gpu_id=1),threshold=i).fit_transform(X,y)
        acc,sen,sps,roc_auc = evalate(X_fschi,y,XGBClassifier(),'xgb_embedd')
        score.append(roc_auc)
    plt.plot(threshold,score)
    plt.legend()
    plt.savefig("./data_eval/xgb_embed.png")
    plt.close()


def Wapper(X,y):
    from sklearn.feature_selection import RFE
    score = []
    for i in range(3,200,5):
        X_wrapper = RFE(XGBClassifier(tree_method='gpu_hist', gpu_id=1), n_features_to_select=i, step=100).fit_transform(X,y)
        acc,sen,sps,roc_auc = evalate(X_wrapper,y,XGBClassifier(),'xgb_wrapper')
        score.append(roc_auc)
    plt.plot(range(1,751,50),score)
    plt.xticks(range(1,751,50))
    plt.legend()
    plt.savefig("./data_eval/xgb_wrapper.png")
    plt.close()



dataset_columns = np.load("./dataset_pff/header.npy", allow_pickle=True)
dataset_columns = np.append(dataset_columns,np.array(['label']))

drug_label = {'KANAMYCIN', 'CAPREOMYCIN', 'RIFAMPICIN', 'ETHIONAMIDE', 'ISONIAZID', 'OFLOXACIN', 'STREPTOMYCIN', 'ETHAMBUTOL', 'PROTHIONAMIDE', 'AMIKACIN', 'CIPROFLOXACIN', 'RIFABUTIN', 'PYRAZINAMIDE', 'MOXIFLOXACIN'}
# drug_label = ['MOXIFLOXACIN']
# CYCLOSERINE 'PARA-AMINOSALICYLIC ACID'  LEVOFLOXACIN 
for drug in drug_label:
    print("****************************************")
    #直接加载转变好的dataset
    print(str(drug)+"   loader dataset ...")
    
    # dataset = pd.read_csv("./dataset_pff/data3/"+str(drug)+".csv")
    dd = np.load("./dataset_pff/data3/"+str(drug)+".npy")
    dataset = pd.DataFrame(dd, columns=dataset_columns)
    
    X_data = dataset[dataset.columns[1:-1]]
    # X_data = X_data.loc[:,~((X_data==0).all())]
    print(X_data.shape)
    label = dataset[dataset.columns[-1]]
    print(label.shape)

    if len(set(label.values))<2:
        print(drug, "=label just 1 class ==pass==")
        continue
    del dataset
    # Filter(X_data,label)
    # Wapper(X_data,label)
    Embedded(X_data,label)
    Filter_p(X_data,label)
  