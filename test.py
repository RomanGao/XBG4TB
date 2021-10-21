import pandas as pd 
import numpy as np

ff_msf_values = np.load("./dataset_pff/ff_msf.npy", allow_pickle=True)
ff_msf_header = np.load("./dataset_pff/ff_msf_header.npy", allow_pickle=True)

ff_msf_df = pd.DataFrame(data=ff_msf_values, columns=ff_msf_header)

# print(ff_msf_df.head)

# hp = {'R': 0, 'S': 1}
# def trans(x):
#     return hp[x]
# ff_msf_df["label"]=ff_msf_df["MSORRES"].apply(trans)

# ff_msf_df = ff_msf_df.drop(["MSORRES","MSREFID" ], axis=1)

# del ff_msf_values
# del ff_msf_header

for drug,group in ff_msf_df.groupby('MSDRUG'):
    print("========", drug, "=======")
    print("length:",len(set(group["ISOLATID"])))
#     print(len(set(group["GENE_NUCHANGE"]))
    if len(set(group["label"]))<2:
        print("drug: ", drug, "=====pass=======")
        continue
        
    header = list(set(group["GENE_NUCHANGE"]))
    np.save("./dataset_pff/drug_data2/"+str(drug) +"_columns.npy", np.array(header))
    
    # feature_df = pd.DataFrame(columns=header, dtype=np.int8)
    # label_df = pd.DataFrame(columns=["label"])  ## label
    features = [[]]
    label =[]
    for iso,gp in ff_msf_df.groupby('ISOLATID'):
#         print(iso)
        # id_df = pd.DataFrame([iso], columns=["label"])
        # label_df = pd.concat([label_df, id_df], axis=0)
        # del id_df
        label.append(gp["label"].values)
        temp = []

        # tmp = np.zeros((1,len(header)), dtype=np.int8)
        for id, head in enumerate(header):
            if id==0:
                continue
            if head in gp["GENE_NUCHANGE"].values:
                temp.append(1)
            else:
                temp.append(0)
    features.append(temp)
    print(len(features),",", len(features[0]))
        # tmp_df = pd.DataFrame(tmp,columns=header, dtype=np.int8)
        # feature_df = pd.concat([feature_df, tmp_df], axis=0)

        # del tmp_df
        # del tmp
    # del header

    np.save("./dataset_pff/drug_data2/"+str(drug) +"_feat.npy" ,features,allow_pickle=True)
    np.save("./dataset_pff/drug_data2/"+str(drug) +"_label.npy", label,allow_pickle=True)
    # label_df.to_csv("./dataset_pff/drug_data2/"+str(drug) +"_label.csv",index=False)
    # feature_df.to_csv("./dataset_pff/drug_data2/"+str(drug) +"_feat.csv",index=False)  # feaure_column
    
    # np.save("./dataset_pff/drug_data2/"+str(drug) +"_feat.npy",feature_df.values)
    # np.save("./dataset_pff/drug_data2/"+str(drug) +"_label.npy", label_df.values)
    # del feature_df
    # del label_df