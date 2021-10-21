'''
求P值  T值
'''

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
        self.feat=feat
#         self.label = pd.read_csv(filename2,header=None)
        self.label=label.values 
        self.label=label.ravel()

    #in all classifiers we run 10 times in each run 20 percent keeps as test set and 80 percent train 
    #from train set 20 percent considers as validation set and remaining for training. The validation set is used
    #for setting the classifier threshold (it is based on classification accuracy)
    #10 fold cross validation and 5 runs
    def ensemble_methods(self,method,drug):
        # for keeping performance (sensitivity, specificity, AUC, accuracy)
        feat = pd.read_csv("./dataset_pff/xgb/"+str(drug)+"_imp.csv")

        muts = feat[feat['importance']>=0.005].name.values.tolist()

        wdata = self.feat[muts]
        wdata = wdata.values #features
        wlabel = self.label #labels
        
        # print("====="+str(method)+"=======")
        if method == "rf":
            
            #models depending which one you wwant to select:
            model = RandomForestClassifier(n_estimators=100, max_depth=None, max_features=1000 )
        elif method == "xgb":
            model = XGBClassifier()
        elif method == "lr":
            model = LogisticRegression()
        else:
            print("model name error !!!!")
        #balancing the data before test and train
        # data,label=self.balanced_subsample(wdata,wlabel)
        

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
        # myDF3["gene_change"],myDF3["feature_importance"],myDF3["Standard Errors"],myDF3["t values"],myDF3["p values"] = [muts,params,sd_b,ts_b,p_values]
        # # print(myDF3)
        # myDF3.to_csv("./stat_res2/"+str(drug)+"_stat.csv", index= False)
        # print(" finish ... ")
        params = model.feature_importances_
        predictions = model.predict(wdata)
        # print(predictions)
        # exit()
        newX = pd.DataFrame(wdata)
        MSE = (sum((wlabel-predictions)**2))/(len(newX)-len(newX.columns))

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
        # myDF3.to_csv("./stat_res/"+str(drug)+"-stat_line.csv", index= False)
        myDF3.to_csv("./stat_res2/"+str(drug)+"_stat.csv", index= False)
        print(" finish ... ")
        # myDF3.close()
        del myDF3


import numpy as np
import pandas as pd

dataset_columns = np.load("./dataset_pff/header.npy", allow_pickle=True)
dataset_columns = np.append(dataset_columns,np.array(['label']))

drug_label = {'KANAMYCIN', 'CAPREOMYCIN', 'RIFAMPICIN', 'ETHIONAMIDE', 'ISONIAZID', 'OFLOXACIN', 'STREPTOMYCIN', 'ETHAMBUTOL', 'PROTHIONAMIDE', 'AMIKACIN', 'CIPROFLOXACIN', 'RIFABUTIN', 'PYRAZINAMIDE', 'MOXIFLOXACIN','CAPREOMYCIN', 'AMIKACIN', 'KANAMYCIN', 'RIFABUTIN', 'CIPROFLOXACIN'}
# drug_label = {'RIFABUTIN', 'CIPROFLOXACIN',}
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