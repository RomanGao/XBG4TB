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

import eli5
from eli5.sklearn import PermutationImportance
from collections import Counter

class mutation_permutation:
    #load data and label
    #filename1 is the feature matrix (rows:isolates and cols: features) and filename2 is labels one for each isolates
    def __init__(self, feat,label):
        self.feat=feat
#         self.label = pd.read_csv(filename2,header=None)
        self.label=label


    #rank mutations based on multilabel learning or single label learnng
    # defualt is multi
    #fname contains the mutation list
    def mutation_importance(self,col,path = ""): 
        
        wdata = self.feat
        wlabel = self.label
        
        
        model = OneVsRestClassifier(XGBClassifier(tree_method='gpu_hist'))
        model.fit(wdata,wlabel)
        print("Feature ranking ...")

        perm = PermutationImportance(model, random_state=1).fit(wdata,wlabel)
        pi_features = eli5.explain_weights_df(perm, feature_names = col)
        pi_features.to_csv(path, index=False)
        
        print("feature stop ...")


print("gene start....")
gene_muti = pd.read_csv("../2_data/gene/multi_feature/MDRv2.csv")
label = ['ISONIAZID','RIFAMPICIN','ETHAMBUTOL','PYRAZINAMIDE']
gene_label = gene_muti[['ISONIAZID','RIFAMPICIN','ETHAMBUTOL','PYRAZINAMIDE']]
gene_feat = gene_muti.drop(['ISONIAZID','RIFAMPICIN','ETHAMBUTOL','PYRAZINAMIDE'], axis =1)

imp_x=mutation_permutation(gene_feat, gene_label)
path = "../3_data/perimp/gene/multi/xgb_permu_imp.csv"
imp_x.mutation_importance(gene_feat.columns, path = path)


print("======================")

print("nu start....")
nu_muti = pd.read_csv("../2_data/nu/multi_feature/MDRv3.csv")
nu_label = nu_muti[['ISONIAZID','RIFAMPICIN','ETHAMBUTOL','PYRAZINAMIDE']]
nu_feat = nu_muti.drop(['ISONIAZID','RIFAMPICIN','ETHAMBUTOL','PYRAZINAMIDE'], axis =1)

imp_x=mutation_permutation(nu_feat, nu_label)
path = "../3_data/perimp/nu/multi/xgb_permu_imp.csv"
imp_x.mutation_importance(nu_feat.columns, path = path)

print("nu finish...")
