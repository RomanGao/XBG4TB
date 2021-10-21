from sklearn.metrics import confusion_matrix
from sklearn import metrics
from model import EnsembleLearning_multilabel
from feature_extraction import Features
# from mutation_ranking import mutation_ranking
import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.model_selection import RandomizedSearchCV
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier,XGBRFClassifier
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold



class mutation_ranking:
    #load data and label
    #filename1 is the feature matrix (rows:isolates and cols: features) and filename2 is labels one for each isolates
    def __init__(self, feat,label):
        self.feat=feat.values
#         self.label = pd.read_csv(filename2,header=None)
        self.label=label.values

    #rank mutations based on multilabel learning or single label learnng
    # defualt is multi
    #fname contains the mutation list
    def mutation_importance(self,col, ddrug,thr = 0.01, vis = True, ncmp = 10):
        
        wdata = self.feat
        wlabel = self.label
        
        #load the mutations list
#         mut = pd.read_csv(fname,header=None)
#         mut = mut.values
#         mut = np.load(fname)
#         imuts = []
        
        #fit the model
#         model = RandomForestClassifier(n_estimators=50, max_depth=None, max_features=None,oob_score=True)
#         model.fit(wdata,wlabel)
#         print("Feature ranking....")
#         imp = pd.DataFrame({"name":col ,"importance":model.feature_importances_})
#         imp["importance_abs"] = imp["importance"].abs()
#         imp = imp.sort_values(by='importance_abs',ascending=False)
#         imp.to_csv("./dataset_pff/data_multi/rf_imp.csv", index=False)
# #         print(imp.head(10))

        
        model = OneVsRestClassifier(XGBClassifier())
        model.fit(wdata,wlabel)
        print("Feature ranking....")
        imp = pd.DataFrame({"name":col ,"importance":model.estimators_[1].feature_importances_})
        imp["importance_abs"] = imp["importance"].abs()
        imp = imp.sort_values(by='importance_abs',ascending=False)
        imp.to_csv("./feature_imp/xgb_"+ str(ddrug)+".csv", index=False)
#         print(imp.head(10))


def combine(temp_list, n):
    '''根据n获得列表中的所有可能组合（n个元素为一组）'''
    temp_list2 = []
    for c in combinations(temp_list, n):
        c = list(c)
        if len(c)>1:
            temp_list2.append(list(c))
    return temp_list2


feat_df = pd.read_csv("./feat_last.csv")
muti_label = pd.read_csv("./label_last.csv")
#3. Classification 

ls =['ISONIAZID', 'RIFAMPICIN', 'ETHAMBUTOL', 'PYRAZINAMIDE']

# end_list = []
# for i in range(1,len(ls)):
#     end_list.extend(combine(ls, i))
# print(end_list)

# for ls in end_list:

print('******** Classfication ********')
x=mutation_ranking(feat_df[feat_df.columns[1:]], muti_label[ls])
#     # print('Ensemble Learning: F1 for INH')
#     # print('******** F1 + RF')
#     x.ensemble_methods('rf', str(drug))
x.mutation_importance(feat_df.columns[1:], '&'.join(ls))