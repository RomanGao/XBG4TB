'''
求P值  T值
'''
from sklearn.multiclass import OneVsRestClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
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
        self.feat=feat.values
#         self.label = pd.read_csv(filename2,header=None)
        self.label=label.values 
        # self.label=label.ravel()

    #in all classifiers we run 10 times in each run 20 percent keeps as test set and 80 percent train 
    #from train set 20 percent considers as validation set and remaining for training. The validation set is used
    #for setting the classifier threshold (it is based on classification accuracy)
    #10 fold cross validation and 5 runs
    def ensemble_methods(self,method,drug,muts):
        # for keeping performance (sensitivity, specificity, AUC, accuracy)
        # accm=[]
        # senm=[]
        # spsm=[]
        # roc_aucm=[]
        # feat = pd.read_csv("./featur_imp/xgb_imp"+str(drug)+".csv")
        # muts = feat.name.values[:60].tolist()
        # #data
        # wdata = self.feat[muts]
        # wdata
        # wdata = wdata.values #features
        # wlabel = self.label #labels
        wdata = self.feat
        wlabel = self.label
        
        # print("====="+str(method)+"=======")
        # if method == "rf":
            
        #     #models depending which one you wwant to select:
        #     model = RandomForestClassifier(n_estimators=100, max_depth=None, max_features=1000 )
        # elif method == "xgb":
        #     model = XGBClassifier()
        # elif method == "lr":
        #     model = LogisticRegression()
        # else:
        #     print("model name error !!!!")
        #balancing the data before test and train
        # data,label=self.balanced_subsample(wdata,wlabel)
        model = OneVsRestClassifier(XGBClassifier(tree_method = 'gpu_hist',gpu_id=1, ))

        model.fit(wdata,wlabel)
        # params = np.append(model.intercept_,model.coef_)

        
        # predictions = model.predict(wdata)

        
        # newX = pd.DataFrame({"Constant":np.ones(len(wdata))}).join(pd.DataFrame(wdata))
        # MSE = (sum((wlabel-predictions)**2))/(len(newX)-len(newX.columns))


        # # Note if you don't want to use a DataFrame replace the two lines above with
        # # newX = np.append(np.ones((len(X),1)), X, axis=1)
        # # MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))
        
        # var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
        # sd_b = np.sqrt(var_b)
        # ts_b = params/ sd_b

        # p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX.columns)-1))) for i in ts_b]

        # sd_b = np.round(sd_b,3)
        # ts_b = np.round(ts_b,3)
        # p_values = np.round(p_values,3)

        # params = np.round(params,4)

        # myDF3 = pd.DataFrame()
        # myDF3["feature_importance"],myDF3["Standard Errors"],myDF3["t values"],myDF3["p values"] = [params,sd_b,ts_b,p_values]
        # # print(myDF3)
        # myDF3.to_csv("./stat_res/"+str(drug)+"_stat.csv", index= False)
        # print(" finish ... ")

        params = model.estimators_[1].feature_importances_
        predictions = model.predict(wdata)
        # print(predictions)
        # exit()
        newX = pd.DataFrame(wdata)
        MSE = sum(sum((wlabel-predictions)**2))/(len(newX)-len(newX.columns))/4

        # Note if you don't want to use a DataFrame replace the two lines above with
        # newX = np.append(np.ones((len(X),1)), X, axis=1)
        # MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

        var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
        sd_b = np.sqrt(var_b)

        ts_b = params/ sd_b

        p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX.columns)-1))) for i in ts_b]

        sd_b = np.round(sd_b,3)
        ts_b = np.round(ts_b,3)
        p_values = np.round(p_values,3)

        params = np.round(params,4)

        myDF3 = pd.DataFrame()
        myDF3["gene_change"],myDF3["feature_importance"],myDF3["Standard Errors"],myDF3["t values"],myDF3["p_values"] = [muts,params,sd_b,ts_b,p_values]
        # print(myDF3)
        myDF3.to_csv("./stat_res/"+str(drug)+"-stat_line.csv", index= False)
        print(" finish ... ")
        # myDF3.close()
        del myDF3


import numpy as np
import pandas as pd

# dataset_columns = np.load("./dataset_pff/header.npy", allow_pickle=True)
# dataset_columns = np.append(dataset_columns,np.array(['label']))

# drug_label = {'KANAMYCIN', 'CAPREOMYCIN', 'RIFAMPICIN', 'ETHIONAMIDE', 'ISONIAZID', 'OFLOXACIN', 'STREPTOMYCIN', 'ETHAMBUTOL', 'PROTHIONAMIDE', 'AMIKACIN', 'CIPROFLOXACIN', 'RIFABUTIN', 'PYRAZINAMIDE', 'MOXIFLOXACIN','CAPREOMYCIN', 'AMIKACIN', 'KANAMYCIN', 'RIFABUTIN', 'CIPROFLOXACIN'}
# drug_label = {'RIFABUTIN', 'CIPROFLOXACIN',}
# AMIKACIN  KANAMYCIN RIFABUTIN CIPROFLOXACIN

drug_label = {
    "ISONIAZID&RIFAMPICIN&ETHAMBUTOL&PYRAZINAMIDE":"F1",
    "ISONIAZID&RIFAMPICIN":"F2",
    "ISONIAZID&ETHAMBUTOL":"F3",
    "ISONIAZID&PYRAZINAMIDE":"F4",
    "RIFAMPICIN&ETHAMBUTOL":"F5",
    "RIFAMPICIN&PYRAZINAMIDE":"F6",
    "ETHAMBUTOL&PYRAZINAMIDE":"F7",
    "ISONIAZID&RIFAMPICIN&ETHAMBUTOL":"F8",
    "ISONIAZID&RIFAMPICIN&PYRAZINAMIDE":"F9",
    "ISONIAZID&ETHAMBUTOL&PYRAZINAMIDE":"F10",
    "RIFAMPICIN&ETHAMBUTOL&PYRAZINAMIDE":"F11",
}

feat_df = pd.read_csv("./feat_last.csv")
muti_label = pd.read_csv("./label_last.csv")

for ls in drug_label.keys():
    print("****************************************")
    #直接加载转变好的dataset
    print(str(ls)+"   loader dataset ...")

    feat = pd.read_csv("./feature_imp/xgb_imp"+ str(ls)+".csv")
    print(str(ls.split('&'))+"==="+str(drug_label[ls]))

    muts = feat[feat['importance']>=0.005].name.values.tolist()

    x = EnsembleLearning(feat_df[muts], muti_label[ls.split("&")])
#     # print('Ensemble Learning: F1 for INH')
#     # print('******** F1 + RF')
    # x.ensemble_methods('rf', str(drug))
    x.ensemble_methods('xgb', str(drug_label[ls]),muts)
    # x.ensemble_methods('ctb', str(drug))