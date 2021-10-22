'''
feature importance 提取
'''
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit
import eli5
from eli5.sklearn import PermutationImportance

from collections import Counter

from sklearn.feature_selection import SelectFromModel

def perf_cal(array1,array2):
    fpr, tpr, thresholds = metrics.roc_curve(array1, array2)
    roc_auc=metrics.auc(fpr, tpr)
    return roc_auc 


def select_thred(X_train,y_train,X_test,y_test ,importances=None, path =None):

     
    model = OneVsRestClassifier(XGBClassifier(tree_method='gpu_hist'))
    model.fit(X_train,y_train)
    
    thresholds = []
    for importance in importances:
        if importance not in thresholds:
            thresholds.append(importance)

    thresholds = sorted(thresholds)
    df = pd.DataFrame(columns = ['thred', 'auc'])
    for threshold in thresholds:
        #进行特征选择
#         for clf in model.estimators_:
        selection = SelectFromModel(model.estimators_[0], threshold  = threshold, prefit = True)
        select_x_train = selection.transform(X_train)
        aucm=np.zeros((4,4))
        kk=0
        #训练模型
        selection_model = OneVsRestClassifier(XGBClassifier(tree_method='gpu_hist'))

        selection_model.fit(select_x_train, y_train)

        #评估模型
        select_x_test = selection.transform(X_test)
        pred = selection_model.predict(select_x_test)
    #     auc = roc_auc_score(y_test, y_pred)
    #     print("阈值： %.5f , 特征数量为：%d, AUC 得分： %.2f" %(threshold,select_x_train.shape[1], auc))
        for j in range(len(pred[0])):
            
            aucm[kk][j]=perf_cal(y_test[:,j],pred[:,j])
            kk=kk+1
            print("--kk--", kk)
    #                 break
            # return 
    #     f = open(path, "w")
    #     f.write("xgb_res ={'accm': np.array("+str((accm*100).tolist())+"),\n'senm':np.array(" + str((senm*100).tolist())+"),\n'spsm': np.array("+str((spsm*100).tolist())+ "),\n'aucm' : np.array("+str((aucm*100).tolist())+")}\n")
    #     f.close()
    #         print("accm: ", accm*100, "\nsenm: ", senm*100, "\nspsm: ",spsm*100, "\naucm:", aucm*100)a
        auc = np.array((aucm*100).tolist()).max(axis = 1).mean().round(2)
        
        df = df.append(pd.Series({"thred":threshold, "auc":auc}),ignore_index=True)
    df.to_csv(path, index =False)

print("gene start....")
gene_muti = pd.read_csv("../2_data/gene/multi_feature/MDRv2.csv")
label = ['ISONIAZID','RIFAMPICIN','ETHAMBUTOL','PYRAZINAMIDE']
gene_label = gene_muti[['ISONIAZID','RIFAMPICIN','ETHAMBUTOL','PYRAZINAMIDE']]
gene_feat = gene_muti.drop(['ISONIAZID','RIFAMPICIN','ETHAMBUTOL','PYRAZINAMIDE'], axis =1)

for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
    print(importance_type)
    from_path = "../3_data/featimp/gene/multi/xgb_"+str(importance_type)+"_imp.csv"
    importances = pd.read_csv(from_path)[importance_type].values
#     print(importances)
    X_train, X_test, y_train, y_test = train_test_split(gene_feat, gene_label, test_size=0.2, random_state=2021)
    to_path =  f'../3_data/thredselect/gene/multi/xgb_{importance_type}_select.csv'
    select_thred(X_train,y_train,X_test,y_test.values ,importances = importances, path = to_path)
    
print("gene finish...")

print("======================")

print("nu start....")
nu_muti = pd.read_csv("../2_data/nu/multi_feature/MDRv3.csv")
nu_label = nu_muti[['ISONIAZID','RIFAMPICIN','ETHAMBUTOL','PYRAZINAMIDE']]
nu_feat = nu_muti.drop(['ISONIAZID','RIFAMPICIN','ETHAMBUTOL','PYRAZINAMIDE'], axis =1)

for importance_type in ('weight', 'gain', 'cover', 'total_gain', 'total_cover'):
    print(importance_type)
    from_path = "../3_data/featimp/nu/multi/xgb_"+str(importance_type)+"_imp.csv"
    importances = pd.read_csv(from_path)[importance_type].values
#     print(importances)
    X_train, X_test, y_train, y_test = train_test_split(nu_feat, nu_label, test_size=0.2, random_state=2021)
    to_path =  f'../3_data/thredselect/nu/multi/xgb_{importance_type}_select.csv'
    select_thred(X_train,y_train,X_test,y_test.values ,importances = importances, path = to_path)

print("nu finish...")
