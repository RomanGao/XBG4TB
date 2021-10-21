#! python
import pandas as pd
import numpy as np
from numpy  import array
##Samaneh Kouchaki
## this file is for extracting features (F1-F5) for multi-label learning of first line drugs (INH, EMB, RIF, PZA, XDR, MDR)
## f1: all variants f2 all mutations based on drug-associated gens
## f3 known features from the litrature
## f4-f5 are all without known isolates
## for f2 and f3 as we are learning data based on multi-label learning all genes/mutations associated to firstline drugs considiered
  
class Features:
    #load snps and phenos for all drugs
    # filename1: name of file with samples information, filename2: pheno, filenames optional for known mutations
    ### 'R' for known ones optional and changable !!! 
    def __init__(self, filename1,filename2,filename3=''): 
       self.feat = pd.read_csv(filename1)#all mutations, insertion, dels
       self.mut = list(self.feat) #header contains all mutations
       self.feat = self.feat.values #all data samples there are several rows per each isolates
       self.pheno = pd.read_csv(filename2) #phenos information for all 11 drugs
       self.druglist =  list(self.pheno) # list of drugs from the pheno file
       #['','SM','KAN','AK','CAP','EMB','CIP','OFX','MOX','INH','RIF','PZA'] 
       self.pheno = self.pheno.values # remove headers
       self.pheno[np.where(pd.isnull(self.pheno))] = -1
       if filename3 != '': #load interesting features (known mutations from litrature)
           # file format should have mutations in first columns, drug name in second line and 'R' or 'S' in third columns 
           lan = pd.read_csv(filename3) #lancet mutations
           lan = lan.values #remove headers
           self.rm = lan#[lan[:,2] == 'R',:] #all with R related for biological methods
           self.um = np.unique(self.rm[:,0]) # remove non unique mutations        
    
    #multi-label (f1) multilabel learning return baseline feature for all first four line drugs
    # name1 to save features (one row per isolates), 
    # name2: to save pheno, 
    #name3: save mutations (this one later used for feature ranking plot), 
    #ind1 is for mdr
    def find_multilabel(self,name1,name2,ind,name3, ind1): 
        #final array for features and labels
        ft = []
        ml = [] 
#        lin = []
        for i in range(len(self.pheno)):
            flag = 1
            label=[]
            # to count if all drugs are resistant to associated drugs for mdr/xdr labels
            s1 = 0 #number of resistant
            s2 = 0 # number of resistant for mdr 
            
            #convert nan to -1 (missing values)
            for j in ind:
                if np.isnan(self.pheno[i][j]):
                    self.pheno[i][j] == -1
                    
            #count to see if the sample is a xdr or mdr
            for j in ind:
                if self.pheno[i][j] == 1: #resistant
                    s1 = s1 + 1
            for j in ind1:
                if self.pheno[i][j] == 1: #resistant
                    s2 = s2 + 1
                    
            # flag =-1 if there is a missing label        
            for j in ind:
                if self.pheno[i][j] == -1:
                    flag = -1
                    break
            
            if flag == 1: #non of the labels are non calls 
                for j in ind:
                    label.append(self.pheno[i][j]) #add drugs labels
                    
                #add mdr and xdr as a separate label
                if s1 == len(ind): #for xdr all first line drugs should be one to get one
                    label.append(1)    
                else:
                    label.append(0)
                    
                if s2 == len(ind1): #if either resistant to atleat INH & RIF 
                    label.append(1)
                else:
                    label.append(0)
                    
                #add features and label to the final matrix   
#                lin.append(self.pheno[i][13])
                ft.append(self.feat[i][1:])
                ml.append(label)
        #save features, labels, mutations (for ranking)        
        np.savetxt(name1, ft, delimiter=",", fmt="%s")
        np.savetxt(name2, ml, delimiter=",", fmt="%s")
        np.savetxt(name3, self.mut[1:], delimiter=",", fmt="%s")
#        np.savetxt('lin.csv', lin, delimiter=",", fmt="%s")
        del ft,ml
        
    #multi-label (f2) multilabel learning return all variants considering drugrelated genes for all first four line drugs and M/XDR
    # name1 to save features (one row per isolates), 
    # name2: to save pheno, 
    #name3: save mutations (this one later used for feature ranking plot), 
    #ind1 is for mdr
    def find_multilabel_drugrelated(self,name1,name2,ind,name3, ind1): 
        #all first line drugs considered and the related gens extracted
        genes = self.suspected_genes('allfirst')
        
        # associated mutation indices found and remaining removed from the featuee space
        # accordingly mutations updated
        inds = self.find_genes(self.mut,genes)       
        feat,mut = self.remove_snps(self.feat,inds)
        
        #final array for features and labels
        ft = []
        ml = [] 
        
        for i in range(len(self.pheno)):
            flag = 1
            label=[]
            # to count if all drugs are resistant to associated drugs for mdr/xdr labels
            s1 = 0 #number of resistant
            s2 = 0 # number of resistant for mdr 
            
            #convert nan to -1 (missing values)
            for j in ind:
                if np.isnan(self.pheno[i][j]):
                    self.pheno[i][j] == -1
            
            #count to see if the sample is a xdr or mdr
            for j in ind:
                if self.pheno[i][j] == 1: #resistant
                    s1 = s1 + 1
            for j in ind1:
                if self.pheno[i][j] == 1:#resistant
                    s2 = s2 + 1
                    
            # flag =-1 if there is a missing label        
            for j in ind:
                if self.pheno[i][j] == -1:
                    flag = -1
                    break
                        
            if flag == 1: #non of the labels are non calls
                for j in ind:
                    label.append(self.pheno[i][j]) #add drugs labels
                    
                #add mdr and xdr as a separate label
                if s1 == len(ind):#for xdr all first line drugs should be one to assign one
                    label.append(1)
                else:
                    label.append(0)
                    
                if s2 == len(ind1): #if either resistant to RIF & INH
                    label.append(1)
                else:
                    label.append(0)
                
                #add features and label to the final matrix        
                ft.append(feat[i][:])
                ml.append(label) 
                
        #save features, labels, mutations (for ranking)        
        np.savetxt(name1, ft, delimiter=",", fmt="%s")
        np.savetxt(name2, ml, delimiter=",", fmt="%s")
        np.savetxt(name3, mut, delimiter=",", fmt="%s")
        del ft,ml
        
    #mutli-label; (f3) known multifeatures for first line drugs
    # name1 to save features (one row per isolates), 
    # name2: to save pheno, 
    #name3: save mutations (this one later used for feature ranking plot), 
    #ind1 is for mdr
    def mutli_Lancet_featurs(self,name1,name2,ind,name3, ind1): #return class labels not features
        #final array for features and labels
        ml = []
        ft = []
        
        #consider all drugs for feature extraction
        drugs = []
        for j in ind:
            drugs.append(self.druglist[j])
            
        for i in range(len(self.pheno)):
            
            flag = 1
            l2 = []
            
            s1 = 0 #number of resistant
            s2 = 0 # for mdr 
            
            #convert nan to -1 (missing values)
            for j in ind:
                if np.isnan(self.pheno[i][j]):
                    self.pheno[i][j] == -1
            
            #count to see if the sample is a xdr or mdr
            for j in ind:
                if self.pheno[i][j] == 1: #resistant
                    s1 = s1 + 1
            for j in ind1:
                if self.pheno[i][j] == 1: #resistant
                    s2 = s2 + 1
            
            # flag =0 if there is a missing label        
            for j in ind:
                if self.pheno[i][j] == -1:
                    flag = -1
                    break
                    
            if flag == 1: #non of the labels are non calls
                for j in ind:
                    l2.append(self.pheno[i][j])#keep samples if resistant/susceptible
                    
                #add mdr and xdr as a separate label
                if s1 == len(ind):#if either resistant to all first line drugs 
                    l2.append(1) 
                else:
                    l2.append(0)
                    
                if s2 == len(ind1): #if either resistant to atleast INH & RIF
                    l2.append(1)
                else:
                    l2.append(0)
                
                #update mutations list to be used for feature ranking
                muts = self.find_snps_multi(self.feat[i][1:],self.mut,drugs)
                ft.append(np.array(muts))
                ml.append(l2)
                
        #save features, labels, mutations (for ranking)      
        np.savetxt(name1, ft, delimiter=",", fmt="%s")
        np.savetxt(name2, ml, delimiter=",", fmt="%s")
        np.savetxt(name3, self.um, delimiter=",", fmt="%s")
        del ml,ft
        
    #multi-label; (f4) multilabel learning return all isolates except for one with known mutations
    # name1 to save features (one row per isolates), 
    # name2: to save pheno, 
    #name3: save mutations (this one later used for feature ranking plot), 
    #ind1 is for mdr
    def find_multilabel_notknown(self,name1,name2,ind,name3, ind1): 
        
        # all drug list to find the related know mutations 
        drugs = []
        for j in ind:
            drugs.append(self.druglist[j])
        #find all known indeces and remove them
        inds=self.find_index_multi_v2(self.mut,drugs)
        
        #final array for features and labels
        ft = []
        ml = [] 
#        lin = []
        for i in range(len(self.pheno)):
            flag = 1
            label=[]
            # to count if all drugs are resistant to associated drugs for mdr/xdr labels
            s1 = 0 #number of resistant
            s2 = 0 # number of resistant for mdr 
            
            #convert nan to -1 (missing values)
            for j in ind:
                if np.isnan(self.pheno[i][j]):
                    self.pheno[i][j] == -1
                    
            #count to see if the sample is a xdr or mdr
            for j in ind:
                if self.pheno[i][j] == 1: #resistant
                    s1 = s1 + 1
            for j in ind1:
                if self.pheno[i][j] == 1: #resistant
                    s2 = s2 + 1
                    
            # flag =-1 if there is a missing label        
            for j in ind:
                if self.pheno[i][j] == -1:
                    flag = -1
                    break
            if np.sum(self.feat[i][inds])>0:
                flag = -1      
            
            if flag == 1: #non of the labels are non calls 
                for j in ind:
                    label.append(self.pheno[i][j]) #add drugs labels
                    
                #add mdr and xdr as a separate label
                if s1 == len(ind): #for xdr all first line drugs should be one to get one
                    label.append(1)    
                else:
                    label.append(0)
                    
                if s2 == len(ind1): #if either resistant to atleat INH & RIF 
                    label.append(1)
                else:
                    label.append(0)
                    
                #add features and label to the final matrix   
#                lin.append(self.pheno[i][13])
                ft.append(self.feat[i][1:])
                ml.append(label)
        #save features, labels, mutations (for ranking)        
        np.savetxt(name1, ft, delimiter=",", fmt="%s")
        np.savetxt(name2, ml, delimiter=",", fmt="%s")
        np.savetxt(name3, self.mut[1:], delimiter=",", fmt="%s")
#        np.savetxt('lin.csv', lin, delimiter=",", fmt="%s")
        del ft,ml
        
    #(multi-label; (f5) multilabel learning return all variants except for known ones
    # name1 to save features (one row per isolates), 
    # name2: to save pheno, 
    #name3: save mutations (this one later used for feature ranking plot), 
    #ind1 is for mdr
    def find_multilabel_dr_notknown(self,name1,name2,ind,name3, ind1): 
        # only keep the associated genes hence update features and mutations
        genes = self.suspected_genes('allfirst')
        inds = self.find_genes(self.mut,genes)       
        f,m = self.remove_snps(self.feat,inds)
        
         # all drug list to find the related know mutations 
        drugs = []
        for j in ind:
            drugs.append(self.druglist[j])
        #find all known indeces and remove them
        inds=self.find_index_multi_v2(m.tolist(),drugs)
        
        #final array for features and labels
        ft = []
        ml = [] 
#        lin = []
        for i in range(len(self.pheno)):
            flag = 1
            label=[]
            # to count if all drugs are resistant to associated drugs for mdr/xdr labels
            s1 = 0 #number of resistant
            s2 = 0 # number of resistant for mdr 
            
            #convert nan to -1 (missing values)
            for j in ind:
                if np.isnan(self.pheno[i][j]):
                    self.pheno[i][j] == -1
                    
            #count to see if the sample is a xdr or mdr
            for j in ind:
                if self.pheno[i][j] == 1: #resistant
                    s1 = s1 + 1
            for j in ind1:
                if self.pheno[i][j] == 1: #resistant
                    s2 = s2 + 1
                    
            # flag =-1 if there is a missing label        
            for j in ind:
                if self.pheno[i][j] == -1:
                    flag = -1
                    break
            if np.sum(f[i][inds])>0:
                flag = -1      
            
            if flag == 1: #non of the labels are non calls 
                for j in ind:
                    label.append(self.pheno[i][j]) #add drugs labels
                    
                #add mdr and xdr as a separate label
                if s1 == len(ind): #for xdr all first line drugs should be one to get one
                    label.append(1)    
                else:
                    label.append(0)
                    
                if s2 == len(ind1): #if either resistant to atleat INH & RIF 
                    label.append(1)
                else:
                    label.append(0)
                    
                #add features and label to the final matrix   
#                lin.append(self.pheno[i][13])
                ft.append(f[i,:])
                ml.append(label)
        #save features, labels, mutations (for ranking)        
        np.savetxt(name1, ft, delimiter=",", fmt="%s")
        np.savetxt(name2, ml, delimiter=",", fmt="%s")
        np.savetxt(name3, m, delimiter=",", fmt="%s")
#        np.savetxt('lin.csv', lin, delimiter=",", fmt="%s")
        del ft,ml
     
    # this function returns the true value and results of direct association
    # name1 to save features (one row per isolates), 
    # name2: to save pheno, 
    #name3: save mutations (this one later used for feature ranking plot), 
    #ind1 is for mdr
    def mutli_Lancet_labels(self,name1,name2,ind,name3, ind1): #return class labels not features
        ml = []
        tl = []
        for i in range(len(self.pheno)):
            flag = 1
            l1 = [] # estimated labels based on known mutations
            l2 = [] # true labels
            s1 = 0 #number of resistant to four first line drugs
            s2 = 0 # for mdr 
            
            #convert nan to -1 (missing values)
            for j in ind:
                if np.isnan(self.pheno[i][j]):
                    self.pheno[i][j]
            
            #count to see if the sample is a xdr or mdr
            for j in ind:
                if self.pheno[i][j] == 1: #resistant
                    s1 = s1 + 1
            for j in ind1:
                if self.pheno[i][j] == 1: #resistant
                    s2 = s2 + 1
                    
            # flag =-1 if there is a missing label        
            for j in ind:
                if self.pheno[i][j] == -1:
                    flag = -1
                    break
                        
            if flag == 1: #non of the labels are non calls
                sss1 = 0
                sss2 = 0
                for j in ind:
                    l2.append(self.pheno[i][j])
                    
                    muts = self.find_snps(self.feat[i][1:],self.mut,self.druglist[j])
                    if (sum(muts)>0): # check if DA predicts the label for all drugs as one
                        l1.append(1)
                        sss1 = sss1 + 1 
                    else:
                        l1.append(0)
                    
                    
                for j in ind1:
                    muts = self.find_snps(self.feat[i][1:],self.mut,self.druglist[j])
                    if (sum(muts)>0):
                        sss2 = sss2 + 1 
        
                #add mdr and xdr as a separate true label
                if s1 == len(ind):
                    l2.append(1)
                else:
                    l2.append(0)
                    
                if s2 == len(ind1): #if either resistant to all drugs or none of drugs 
                    l2.append(1)
                else:
                    l2.append(0)
                
                #add mdr and xdr as a separate estimated labels
                if sss1 == len(ind): 
                    l1.append(1)
                else:
                    l1.append(0)
                    
                if sss2 == len(ind1):
                    l1.append(1)
                else:
                    l1.append(0)
                    
                ml.append(l1)
                tl.append(l2)
        #Save true, estimated labels and mutations
        np.savetxt(name1, ml, delimiter=",", fmt="%s")
        np.savetxt(name2, tl, delimiter=",", fmt="%s")
        np.savetxt(name3, self.um, delimiter=",", fmt="%s")
        del ml,tl
                
#########################################
    #multi-label (f2) multilabel learning return all variants considering drugrelated genes for each four line drugs and M/XDR
    # name1 to save features (one row per isolates), 
    # name2: to save pheno, 
    #name3: save mutations (this one later used for feature ranking plot), 
    #ind1 is for mdr
    def find_multilabel_drugrelated_v2(self,name1,name2,ind,name3, ind1,drug): 
        #all first line drugs considered and the related gens extracted
        genes = self.suspected_genes(drug)
        
        # associated mutation indices found and remaining removed from the featuee space
        # accordingly mutations updated
        inds = self.find_genes(self.mut,genes)       
        feat,mut = self.remove_snps(self.feat,inds)
        
        #final array for features and labels
        ft = []
        ml = [] 
        
        for i in range(len(self.pheno)):
            flag = 1
            label=[]
            # to count if all drugs are resistant to associated drugs for mdr/xdr labels
            s1 = 0 #number of resistant
            s2 = 0 # number of resistant for mdr 
            
            #convert nan to -1 (missing values)
            for j in ind:
                if np.isnan(self.pheno[i][j]):
                    self.pheno[i][j] == -1
            
            #count to see if the sample is a xdr or mdr
            for j in ind:
                if self.pheno[i][j] == 1: #resistant
                    s1 = s1 + 1
            for j in ind1:
                if self.pheno[i][j] == 1:#resistant
                    s2 = s2 + 1
                    
            # flag =-1 if there is a missing label        
            for j in ind:
                if self.pheno[i][j] == -1:
                    flag = -1
                    break
                        
            if flag == 1: #non of the labels are non calls
                for j in ind:
                    label.append(self.pheno[i][j]) #add drugs labels
                    
                #add mdr and xdr as a separate label
                if s1 == len(ind):#for xdr all first line drugs should be one to assign one
                    label.append(1)
                else:
                    label.append(0)
                    
                if s2 == len(ind1): #if either resistant to RIF & INH
                    label.append(1)
                else:
                    label.append(0)
                
                #add features and label to the final matrix        
                ft.append(feat[i][:])
                ml.append(label) 
                
        #save features, labels, mutations (for ranking)        
        np.savetxt(name1, ft, delimiter=",", fmt="%s")
        np.savetxt(name2, ml, delimiter=",", fmt="%s")
        np.savetxt(name3, mut, delimiter=",", fmt="%s")
        del ft,ml
        
    #mutli-label; (f3) known multifeatures for first line drugs
    # name1 to save features (one row per isolates), 
    # name2: to save pheno, 
    #name3: save mutations (this one later used for feature ranking plot), 
    #ind1 is for mdr
    def mutli_Lancet_featurs_v2(self,name1,name2,ind,name3, ind1,drugs): #return class labels not features
        #final array for features and labels
        ml = []
        ft = []
        
        #consider all drugs for feature extraction
#        drugs = []
#        for j in ind:
#            drugs.append(self.druglist[j])
            
        for i in range(len(self.pheno)):
            
            flag = 1
            l2 = []
            
            s1 = 0 #number of resistant
            s2 = 0 # for mdr 
            
            #convert nan to -1 (missing values)
            for j in ind:
                if np.isnan(self.pheno[i][j]):
                    self.pheno[i][j] == -1
            
            #count to see if the sample is a xdr or mdr
            for j in ind:
                if self.pheno[i][j] == 1: #resistant
                    s1 = s1 + 1
            for j in ind1:
                if self.pheno[i][j] == 1: #resistant
                    s2 = s2 + 1
            
            # flag =0 if there is a missing label        
            for j in ind:
                if self.pheno[i][j] == -1:
                    flag = -1
                    break
                    
            if flag == 1: #non of the labels are non calls
                for j in ind:
                    l2.append(self.pheno[i][j])#keep samples if resistant/susceptible
                    
                #add mdr and xdr as a separate label
                if s1 == len(ind):#if either resistant to all first line drugs 
                    l2.append(1) 
                else:
                    l2.append(0)
                    
                if s2 == len(ind1): #if either resistant to atleast INH & RIF
                    l2.append(1)
                else:
                    l2.append(0)
                
                #update mutations list to be used for feature ranking
                muts = self.find_snps_multi(self.feat[i][1:],self.mut,drugs)
                ft.append(np.array(muts))
                ml.append(l2)
                
        #save features, labels, mutations (for ranking)      
        np.savetxt(name1, ft, delimiter=",", fmt="%s")
        np.savetxt(name2, ml, delimiter=",", fmt="%s")
        np.savetxt(name3, self.um, delimiter=",", fmt="%s")
        del ml,ft
    
    
    #find unrelated genes to delete them i.e. f3
    def find_genes(self, array, genes):
        inds = []
        #find their indices in the whole feature space        
        for i in range(len(array)):
            flag = 0 #a flag
            for j in range(len(genes)): #for all desired genes list
                if(array[i].find(genes[j]) >= 0): #check if it is from the same gene
                    flag = 1
                    break
            if(flag == 0): # if genes are different keep index to be removed later
                inds.append(i)
        return inds
    
    #returns the list of suspected gens to be assoicated with resistance of a given drug    
    ### I may make it possible to change the genes
    def suspected_genes(self, drug):
        if drug == 'INH':
            genes = ['ahpc','fabG1','inhA','katG','ndh']
        elif drug == 'RIF':
            genes = ['rpoB']
        elif drug == 'EMB':
            genes = ['embA','embB','embC','embR','iniA','iniC','manB','rmlD']
        elif drug == 'PZA':
            genes = ['pncA','rpsA']
        elif drug == 'OFX' or drug == 'MOX' or drug == 'CIP':
            genes = ['gyrA','gyrB']
        elif drug == 'SM':
            genes = ['rpsL','gidB','rrs','tlyA']
        elif drug == 'AK' or drug == 'CAP':
            genes = ['gidB','rrs','tlyA']
        elif drug == 'KAN':
            genes = ['eis','gidB','rrs','tlyA']
        elif drug == 'allfirst': # for INH, EMB, RIF, PZA all together
                genes = ['ahpc','embA','embB','embC','embR','fabG1','inhA','iniC','iniA','katG','ndh','rpoB','manB','rmlD','pcnA','rpsA']
        elif drug == 'mdr': #INH & RIF
            genes = ['ahpc','fabG1','inhA','katG','ndh','rpoB']
        else:
            genes = []
        return genes
    
    #find mutations from the list of known genes
    # array1 is the all mutations for the isolates and array2 the list of corresponding variants, drug is the drug name
    def find_snps(self, array1, array2, drug):
        rm=self.rm
        um=self.um
        res = np.zeros(len(um)) # an array zero that be filled if any known mutation happened in the isolate
        try:
            for i in range(len(rm)):
                if rm[i,1] == drug:
                    if rm[i,2] == 'R':
                        ind = array2.index(rm[i,0])
                        if array1[ind-1] == 1:
                            ind = um.tolist().index(rm[i,0])
                            res[ind] = 1
        except ValueError as e:
            print(e)
        return res
    
    #find mutations from the list of known genes for all drugs in the list
    # array1 is the all mutations for the isolates and array2 the list of corresponding variants, drugs is the drugs list
    def find_snps_multi(self, array1, array2, drugs):
        rm=self.rm
        um=self.um
        res = np.zeros(len(um))
        try:
            for i in range(len(rm)):
                for j in drugs:
                    if rm[i,1] == j:
                        if rm[i,2] == 'R':
                            ind = array2.index(rm[i,0])
                            if array1[ind-1] == 1:
                                ind = um.tolist().index(rm[i,0])
                                res[ind] = 1
        except ValueError as e:
            print(e)
        return res
    
    #finding related indeces to be removed from the feature space based on known mutation library
    def find_index_multi(self,array1,druglist):
        inds=[]
        #find their indices in the whole feature space     
        try:
            for i in range(len(self.rm)):
                for j in druglist:
                    if(self.rm[i][1] == j):
                        if self.rm[i][2] == 'R':
                            x=np.intersect1d(array1,self.rm[i,0])
                            if len(x) > 0:
                                ind = array1.index(self.rm[i][0])
                                if ind != len(array1):
                                    inds.append(ind)
                                    break
        except ValueError as e:
            print(e)
        return inds
    
    #finding related indeces to be removed from the feature space based on known mutation library
    def find_index_multi_v2(self,array1,druglist):
        inds=[]
        #find their indices in the whole feature space     
        try:
            for i in range(len(self.rm)):
                for j in druglist:
                    if(self.rm[i][1] == j):
                        if self.rm[i][2] !='S' and self.rm[i][2] != 'U':
                            x=np.intersect1d(array1,self.rm[i,0])
                            if len(x) > 0:
                                ind = array1.index(self.rm[i][0])
                                inds.append(ind)
        except ValueError as e:
            print(e)
        return inds
    
    #remove some columns indicated by indecies
    # inds is the list of coloumns to be removed from array1
    # it also updates the mutation list
    def remove_snps(self, array1, inds):
        array1 = np.delete(array(array1), inds, 1)
        newmut= np.delete(array(self.mut), inds)
        return array1,newmut

