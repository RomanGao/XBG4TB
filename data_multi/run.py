from model import EnsembleLearning_multilabel
from feature_extraction import Features
import numpy as np
import pandas as pd
from itertools import combinations


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
# #3. Classification 

ls =['ISONIAZID', 'RIFAMPICIN', 'ETHAMBUTOL', 'PYRAZINAMIDE']
end_list = []
for i in range(1,len(ls)):
    end_list.extend(combine(ls, i))
print(end_list)
end_list.append(ls)
print(end_list)
print(len(end_list))


name_dic = {
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


# all feature train and predict
for ls in end_list:
    print('********train and prediction Classfication ********')
    #(DA)###################################### DA   
    x = EnsembleLearning_multilabel(feat_df[feat_df.columns[1:]], muti_label[ls])
    # print('########### DA')
    # accm, senm, specm, aucm = x.DA() 
    # file = open("DA.txt","a")
    # for i in range(len(accm[0])): 
    #     file.write("%f $\pm$ %f & %f $\pm$ %f & %f $\pm$ %f & %f $\pm$ %f \n" % (np.mean(accm,axis = 0)[i], np.std(accm,axis = 0)[i], np.mean(senm,axis = 0)[i], np.std(senm,axis = 0)[i], np.mean(specm,axis = 0)[i], np.std(specm,axis = 0)[i], np.mean(aucm,axis = 0)[i], np.std(aucm,axis = 0)[i]))
    # file.close()
    # #first line drugs

    #1A classification
    print('### MLRF ###')
    print(ls)
    accm, senm, specm, aucm = x.ensemble_multilabel(ls)   #rf model
    # print("accm: ",accm)
    # print("senm: ",senm)
    # print("specm: ",specm)
    # print("aucm: ", aucm)
    file = open('./res/rf'+name_dic["&".join(ls)]+'.txt',"a+")
    for i in range(len(accm[0])): 
        # file.write("%f $\pm$ %f & %f $\pm$ %f & %f $\pm$ %f & %f $\pm$ %f \n" % (accm,axis = 0)[i], np.std(accm,axis = 0)[i], np.mean(senm,axis = 0)[i], np.std(senm,axis = 0)[i], np.mean(specm,axis = 0)[i], np.std(specm,axis = 0)[i], np.mean(aucm,axis = 0)[i], np.std(aucm,axis = 0)[i]))
        file.write("xgb_res ={'accm': np.array("+str((accm).tolist())+"),\n'senm':np.array(" + str((senm).tolist())+"),\n'spsm': np.array("+str((specm).tolist())+ "),\n'aucm' : np.array("+str((aucm).tolist())+")}\n")
    file.close()




# top n of feature importance

# name_dic = {
#     "ISONIAZID&RIFAMPICIN&ETHAMBUTOL&PYRAZINAMIDE":"F1",
#     "ISONIAZID&RIFAMPICIN":"F2",
#     "ISONIAZID&ETHAMBUTOL":"F3",
#     "ISONIAZID&PYRAZINAMIDE":"F4",
#     "RIFAMPICIN&ETHAMBUTOL":"F5",
#     "RIFAMPICIN&PYRAZINAMIDE":"F6",
#     "ETHAMBUTOL&PYRAZINAMIDE":"F7",
#     "ISONIAZID&RIFAMPICIN&ETHAMBUTOL":"F8",
#     "ISONIAZID&RIFAMPICIN&PYRAZINAMIDE":"F9",
#     "ISONIAZID&ETHAMBUTOL&PYRAZINAMIDE":"F10",
#     "RIFAMPICIN&ETHAMBUTOL&PYRAZINAMIDE":"F11",
# }

# thresholds = [0.001, 0.005, 0.01, 0.05] 

# for th in thresholds:
    
#     for ls in list(name_dic.keys()):
#         print('******** top perform of classification ********')
        
#         feat = pd.read_csv("./feature_imp/xgb_imp"+str(ls)+".csv")

#         muts = feat[feat['importance']>=th].name.values.tolist()
#         print(str(ls.split('&'))+"==="+str(name_dic[ls]))
#         x = EnsembleLearning_multilabel(feat_df[muts], muti_label[ls.split("&")])
#         # print('########### DA')
#         # accm, senm, specm, aucm = x.DA() 
#         # file = open("DA.txt","a")
#         # for i in range(len(accm[0])): 
#         #     file.write("%f $\pm$ %f & %f $\pm$ %f & %f $\pm$ %f & %f $\pm$ %f \n" % (np.mean(accm,axis = 0)[i], np.std(accm,axis = 0)[i], np.mean(senm,axis = 0)[i], np.std(senm,axis = 0)[i], np.mean(specm,axis = 0)[i], np.std(specm,axis = 0)[i], np.mean(aucm,axis = 0)[i], np.std(aucm,axis = 0)[i]))
#         # file.close()
#         # #first line drugs

#         #1A classification
#         print('#=====# MLRF #======#')
       
#         accm, senm, specm, aucm = x.ensemble_multilabel2(ls)  #xgb model
#         # print("accm: ",accm)
#         # print("senm: ",senm)
#         # print("specm: ",specm)
#         # print("aucm: ", aucm)
#         file = open('./res/xgb_'+str(th)+'_'+name_dic[ls]+'.txt',"a+")
#             # file.write("%f $\pm$ %f & %f $\pm$ %f & %f $\pm$ %f & %f $\pm$ %f \n" % (accm,axis = 0)[i], np.std(accm,axis = 0)[i], np.mean(senm,axis = 0)[i], np.std(senm,axis = 0)[i], np.mean(specm,axis = 0)[i], np.std(specm,axis = 0)[i], np.mean(aucm,axis = 0)[i], np.std(aucm,axis = 0)[i]))
#         file.write('===xgb_'+str(th)+'_'+name_dic[ls]+'===\n')
#         file.write("xgb_res ={'accm': np.array("+str((accm).tolist())+"),\n'senm':np.array(" + str((senm).tolist())+"),\n'spsm': np.array("+str((specm).tolist())+ "),\n'aucm' : np.array("+str((aucm).tolist())+")}\n")
#         file.close()
