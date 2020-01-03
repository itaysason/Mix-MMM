import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
import json
from os.path import join

if __name__ == "__main__":
    method = "Mix"
    msk_dir ="/Users/yuexichen/Downloads/lrgr_file/mskfiles"
    part_ls = [2]
    ds_ls = [2,5,10]
    for dl in ds_ls:
        doc_f = join(msk_dir, "/Users/yuexichen/Downloads/lrgr_file/mskfiles/part1-status-new-OV-ds%d-OV-ds%d-survival-analysis-assignments-0.tsv"%(dl,dl))
        doc_df = pd.read_csv(doc_f, sep='\t')
        mut_status = list(doc_df['status'])
        mut_id = list(doc_df['id'])
        bina_mut = []
        for ms in mut_status:
            if ms == "BRCA mutations":
                bina_mut.append(1)
            else:
                bina_mut.append(0)
        id_mut_dict = dict(zip(mut_id, bina_mut))
        #print(id_mut_dict)
        #get a well-ordered id 
        ds_id = pd.read_csv(join(msk_dir, "mix_downsize/ov-downsize%d_sample_id.txt"%dl),sep='\n', header=None).values.tolist()
        mut_dict = {}
        for d in range(len(ds_id)):
            #use numerical values as keys
            if ds_id[d][0] in id_mut_dict.keys():
                mut_dict[d]= id_mut_dict[ds_id[d][0]]
            else:
                mut_dict[d]= "na" 
        #print(mut_dict)
        index_all = np.load("/Users/yuexichen/Downloads/lrgr_file/mskfiles/mix_downsize/indices/OV-ds%d-indices.npy"%dl)
        for p in part_ls:
            if method == "Mix":
                expo_f = join(msk_dir, "mix_downsize/models/OV-ds%d-part%d-exposure.npy"%(dl,p))
                expo_np = np.load(expo_f)
                all_y_score = [item[1]/sum(item) for item in expo_np]
            elif method == "SigMA":
                expo_f = join(msk_dir,"mix_downsize/sigma_downsize%d_sig3_part%d.npy"%(dl,p))
                expo_np = np.load(expo_f)
                all_y_score = expo_np
        
            if p == 1:
                index_list =  index_all[:len(index_all) // 2] 
            elif p==2:
                index_list = index_all[len(index_all) // 2:] 
            #y score is already indexed   
            y_score = np.array(all_y_score) 
            y_true =  []
            for il in index_list:
                y_true.append(mut_dict[il])
            new_y_true=[]
            new_y_score=[]
            for i in range(len(y_true)):
                # only keep known brca status
                if y_true[i] != "na":
                    new_y_score.append(y_score[i])
                    new_y_true.append(y_true[i])
            print(len(new_y_true), len(new_y_score))
            ap = average_precision_score(new_y_true, new_y_score) 
            auc = roc_auc_score(new_y_true, new_y_score)
            print("At part %d, the AUPRC= %0.3f, AUROC=%0.3f"%(p, ap, auc))
