import pandas as pd
import numpy as np
def readcsv(name, itera=True,chunkSize=1000):
    reader = pd.read_csv(name, iterator=itera)
    loop = True
    chunks = []
    index = 0
    while loop:
        index += 1
        print(index)
        try:
            print(chunkSize)
            chunk = reader.get_chunk(chunkSize)
            chunks.append(chunk)
            del chunk
        except StopIteration:
            loop = False
            print ("Iteration is stopped.")

    df = pd.concat(chunks, ignore_index=True)
    return df
# print("load msf..")
# msf_RS= pd.read_csv("./dataset_pff/msf_label.csv")

print("load id_feat..")
id_feat = readcsv("./dataset_pff/id_feat.csv")
print("save id_feat columns..")
np.save("./dataset_pff/id_feat_column.npy",id_feat.columns)
print("save id_feat values..")
np.save("./dataset_pff/id_feat.npy", id_feat.values)



# print("..run..")
# for drug,group in msf_RS.groupby(["MSDRUG"]):
#     drug = str(drug).replace("/", "__")
#     print(drug)
#     if len(set(group["label"].values))<2:
#         print("==pass==")
#         continue
#     tmp_df = id_feat[id_feat["ISOLATID"].isin(group["MSREFID"])]
#     res = pd.merge(tmp_df, group, left_on="ISOLATID", right_on="MSREFID")
#     res = res.drop(['ISOLATID','MSREFID',"MSDRUG"], axis=1)
#     print(res.shape)
# #     print(res.head())
#     np.save("./dataset_pff/data3/"+str(drug)+".npy", res.values)
#     res.to_csv("./dataset_pff/data3/"+str(drug)+".csv", index=False)
#     del res