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

import eli5
from eli5.sklearn import PermutationImportance

class mutation_ranking:
    #load data and label
    #filename1 is the feature matrix (rows:isolates and cols: features) and filename2 is labels one for each isolates
    def __init__(self, feat,label):
        self.feat=feat.values
#         self.label = pd.read_csv(filename2,header=None)
        self.label=label
        self.label=label


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
        # model = RandomForestClassifier(n_estimators=50, max_depth=None, max_features=None,oob_score=True)
        # model.fit(wdata,wlabel)
        # print("Feature ranking....")
        # imp = pd.DataFrame({"name":col ,"importance":model.feature_importances_})
        # imp["importance_abs"] = imp["importance"].abs()
        # imp = imp.sort_values(by='importance_abs',ascending=False)
        # imp.to_csv("/htju/gaocl/result_exp/3_data/featimp/nu/single/"+str(drug)+"_rf_imp.csv", index=False)
        # print(imp.head(10))
        
        model = XGBClassifier(tree_method='gpu_hist')
        model.fit(wdata,wlabel)
        print("Feature ranking....")
        imp = pd.DataFrame({"name":col ,"importance":model.feature_importances_})
        imp = imp.sort_values(by='importance',ascending=False)
        imp.to_csv("/htju/gaocl/result_exp/3_data/featimp/nu/single/"+str(drug)+"_xgb_impvsperm.csv", index=False)

        perm = PermutationImportance(model, random_state=1).fit(wdata,wlabel)
        pi_features = eli5.explain_weights_df(perm, feature_names = col.tolist())
        pi_features.to_csv("/htju/gaocl/result_exp/3_data/perimp/nu/single/"+str(drug)+"_xgb_perm.csv", index=False)

drug_label = {'KANAMYCIN', 'CAPREOMYCIN', 'RIFAMPICIN', 'ETHIONAMIDE', 'ISONIAZID', 'OFLOXACIN', 'STREPTOMYCIN', 'ETHAMBUTOL', 'PROTHIONAMIDE', 'AMIKACIN', 'CIPROFLOXACIN', 'RIFABUTIN', 'PYRAZINAMIDE', 'MOXIFLOXACIN'}
for drug in drug_label:
    dataset_columns = np.load("/htju/gaocl/result_exp/2_data/nu/single_feature2/"+drug+"_nuheader.npy", allow_pickle=True)
    # dataset_columns = np.append(dataset_columns,np.array(['label']))
    print("****************************************")
    #直接加载转变好的dataset
    print(str(drug)+"   loader dataset ...")
    
    # dataset = pd.read_csv("./dataset_pff/data3/"+str(drug)+".csv")
    dd = np.load("/htju/gaocl/result_exp/2_data/nu/single_feature2/"+drug+"_nu.npy")
    dataset = pd.DataFrame(dd, columns=dataset_columns)
    

    label = dataset[['label']]
    print(label.shape)
    X_data = dataset.drop(['label'], axis=1)
    # X_data = X_data.loc[:,~((X_data==0).all())]
    print(X_data.shape)

    from_path = "../3_data/featimp/nu/single/"+str(drug)+"_xgb_imp.csv"
    mut = pd.read_csv(from_path)
    thred = 0.001
    mut = mut[mut.importance>=thred].name.values.tolist()
    del dataset
    X_data = X_data[mut]
    x=mutation_ranking(X_data, label)
#     # print('Ensemble Learning: F1 for INH')
#     # print('******** F1 + RF')
#     x.ensemble_methods('rf', str(drug))
    x.mutation_importance(X_data.columns, str(drug))