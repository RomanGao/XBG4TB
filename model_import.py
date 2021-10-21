'''
feature importance 提取
'''
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


class mutation_ranking:
    #load data and label
    #filename1 is the feature matrix (rows:isolates and cols: features) and filename2 is labels one for each isolates
    def __init__(self, feat,label):
        self.feat=feat.values
#         self.label = pd.read_csv(filename2,header=None)
        self.label=label.values
        self.label=label.ravel()


    #rank mutations based on multilabel learning or single label learnng
    # defualt is multi
    #fname contains the mutation list
    def mutation_importance(self,col, figname, thr = 0.01, vis = True, ncmp = 10):
        
        wdata = self.feat
        wlabel = self.label
        
        #load the mutations list
#         mut = pd.read_csv(fname,header=None)
#         mut = mut.values
#         mut = np.load(fname)
#         imuts = []
        
        #fit the model
        model = RandomForestClassifier(n_estimators=50, max_depth=None, max_features=None,oob_score=True)
        model.fit(wdata,wlabel)
        print("Feature ranking....")
        imp = pd.DataFrame({"name":col ,"importance":model.feature_importances_})
        imp["importance_abs"] = imp["importance"].abs()
        imp = imp.sort_values(by='importance_abs',ascending=False)
        imp.to_csv("./dataset_pff/rf/"+str(drug)+"_imp.csv", index=False)
#         print(imp.head(10))
        
        model = XGBClassifier()
        model.fit(wdata,wlabel)
        print("Feature ranking....")
        imp = pd.DataFrame({"name":col ,"importance":model.feature_importances_})
        imp["importance_abs"] = imp["importance"].abs()
        imp = imp.sort_values(by='importance_abs',ascending=False)
        imp.to_csv("./dataset_pff/xgb/"+str(drug)+"_imp.csv", index=False)
#         print(imp.head(10))

dataset_columns = np.load("./dataset_pff/header.npy", allow_pickle=True)
dataset_columns = np.append(dataset_columns,np.array(['label']))
# drug_label = {'KANAMYCIN', 'RIFAMPICIN', 'ETHIONAMIDE', 'ISONIAZID', 'OFLOXACIN', 'STREPTOMYCIN', 'ETHAMBUTOL', 'PROTHIONAMIDE', 'AMIKACIN', 'CIPROFLOXACIN', 'RIFABUTIN', 'PYRAZINAMIDE', 'MOXIFLOXACIN'}
drug_label = {'CAPREOMYCIN'}
# CYCLOSERINE 'PARA-AMINOSALICYLIC ACID' MOXIFLOXACIN LEVOFLOXACIN , 'CAPREOMYCIN'
# drug_label = ['MOXIFLOXACIN']
for drug in drug_label:
    print("****************************************")
    #直接加载转变好的dataset
    print(str(drug)+"   loader dataset ...")
    
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
    x=mutation_ranking(X_data,label)
#     # print('Ensemble Learning: F1 for INH')
#     # print('******** F1 + RF')
#     x.ensemble_methods('rf', str(drug))
    x.mutation_importance(X_data.columns, str(drug))