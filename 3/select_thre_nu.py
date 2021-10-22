import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectFromModel

def func(X, y, drug,path):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2021)

    xgb_clf = xgb.XGBClassifier(
        booster='gbtree',
        objective='binary:logistic',
        gamma= 0.1,
        max_depth= 6,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight= 3,
        eta= 0.1,
        seed=1000,
        nthread= 4,
        gpu_id=0,
        # tree_method='gpu_hist'
    )

    xgb_clf.fit(X_train, y_train)

    # y_pred = xgb_clf.predict(X_test)
    # auc = roc_auc_score(y_test, y_pred)
    # print("AUC 得分： %.2f" %(auc))

    importances = xgb_clf.feature_importances_  # 相当于get_fscore

    thresholds = []
    for importance in importances:
        if importance not in thresholds:
            thresholds.append(importance)

    thresholds = sorted(thresholds)

    res = pd.DataFrame(columns=["thredshold", "featNum", "auc"])

    for threshold in thresholds:
        #进行特征选择
        selection = SelectFromModel(xgb_clf, threshold  = threshold, prefit = True)
        select_x_train = selection.transform(X_train)
        
        #训练模型
        selection_model = xgb.XGBClassifier(
        booster='gbtree',
        objective='binary:logistic',
        gamma= 0.1,
        max_depth= 6,
        subsample=0.7,
        colsample_bytree=0.7,
        min_child_weight= 3,
        eta= 0.1,
        seed=1000,
        nthread= 4,
        gpu_id=0,
        # tree_method='gpu_hist'
        )
        
        selection_model.fit(select_x_train, y_train)
        
        #评估模型
        select_x_test = selection.transform(X_test)
        y_pred = selection_model.predict(select_x_test)
        auc = roc_auc_score(y_test, y_pred)
        res = res.append(pd.Series({"thredshold":threshold,"featNum":select_x_train.shape[1],"auc":auc}),ignore_index=True)
        # print("阈值： %.5f , 特征数量为：%d, AUC 得分： %.2f" %(threshold,select_x_train.shape[1], auc))
    res.to_csv(path, index=False)

drug_label = {'KANAMYCIN', 'CAPREOMYCIN', 'RIFAMPICIN', 'ETHIONAMIDE', 'ISONIAZID', 'OFLOXACIN', 'STREPTOMYCIN', 'ETHAMBUTOL', 'PROTHIONAMIDE', 'AMIKACIN', 'CIPROFLOXACIN', 'RIFABUTIN', 'PYRAZINAMIDE', 'MOXIFLOXACIN'}
# drug_label = ['MOXIFLOXACIN']
# CYCLOSERINE 'PARA-AMINOSALICYLIC ACID'  LEVOFLOXACIN 
for drug in drug_label:

    dataset_columns = np.load("/htju/gaocl/result_exp/2_data/nu/single_feature2/"+drug+"_nuheader.npy", allow_pickle=True)
    # dataset_columns = np.append(dataset_columns,np.array(['label']))
    print("****************************************")
    #直接加载转变好的dataset
    print(str(drug)+"   loader dataset ...")
    
    # dataset = pd.read_csv("./dataset_pff/data3/"+str(drug)+".csv")
    dd = np.load("/htju/gaocl/result_exp/2_data/nu/single_feature2/"+drug+"_nu.npy")
    dataset = pd.DataFrame(dd, columns=dataset_columns)
    
    X_data = dataset[dataset.columns[0:-1]]
    # X_data = X_data.loc[:,~((X_data==0).all())]
    print(X_data.shape)
    label = dataset[dataset.columns[-1]]
    print(label.shape)

    func(X_data, label, str(drug), "/htju/gaocl/result_exp/3_data/thredselect/nu/"+drug+".csv")