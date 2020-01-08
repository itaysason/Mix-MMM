from os.path import join
import json
import pandas as pd
import numpy as np
import re

def concat_expo(select_df_f, npy_f, new_df_f, expo_type):
    """
    input:
    1. the selected dataframe
    2. the numpy file of exposures or assignments
    3. the output file name
    4. the expo_type, exposures and assignments
    output:
    A new dataframe with the new columns
    column 1: binary exposures of signature 3
    column 2: continous exposures of signature 3
    """
    select_df = pd.read_csv(select_df_f, sep='\t')
    all_npy = np.load(npy_f, allow_pickle=True)
   
    bina_item = []
    zero_cnt = 0
    if expo_type == "exposures":
        expo_item = [items[0] for items in all_npy]
    else:
        expo_item = [items[0]/(sum(items)+1e-9) for items in all_npy]

    for expo in expo_item:
        if expo == 0:
            bina_item.append(0)
            zero_cnt += 1
        else:
            bina_item.append(1)
    print("We have %d zero counts"%zero_cnt)
    print(len(bina_item))
    print(len(select_df.index))
    select_df['binary_sig3'] = bina_item
    select_df['exposure_sig3'] = expo_item
    select_df.to_csv(new_df_f, sep='\t', index=None)


def select_df(msk_dir, all_df_f, id_list_f, select_df_f):
    """
    input: 
    1. the whole table of survival dataframe
    2. the patients id list
    output:
    dataframes of rows selected by this id list
    """
    all_df = pd.read_csv(all_df_f, sep=',')
    with open(id_list_f) as in_f:
        id_list = [item.rstrip() for item in in_f.readlines()]
    #enforce the order
    row_ls = []
    for il in id_list:
        now_row = all_df.loc[all_df['tumor'] == il]
        row_ls.append(now_row)

    select_df = pd.concat(row_ls)
    select_df.to_csv(select_df_f, sep=',', index=None)


def row2df(msk_dir, raw_f, out_csv):
    """
    input: raw firehouse file and its directory
    output: a csv file with columns = ["id","survival_status","survival_days", "age_days", "clinical_stage"]
    column explantions
    id: TCGA id
    survival_status: 0: dead 1: alive
    survival_days: days survived until the end of the study (how to deal with NA?)
    age_days: in the unit of days
    clinical_stage: in the scale of numbers, check the possible options first
    """
    with open(all_clinical) as in_f:
        tmp = [x.strip().split('\t') for x in in_f.read().splitlines()]
    for lists in tmp:
        if lists[0] == "bcr_patient_barcode":
            id_ls = lists[1:]
            #capticalize
            id_ls = [items.upper() for items in id_ls]
        elif lists[0] == "vital_status":
            surv_stat_ls = lists[1:]
            nume_stat_ls = []
            for i in range(len(surv_stat_ls)):
                ssl = surv_stat_ls[i]
                if ssl == "dead":
                    nume_stat_ls.append(2)
                elif ssl == "alive":
                    nume_stat_ls.append(1)
                elif ssl == "NA":
                    nume_stat_ls.append(0)
                else:
                    Warning("patients %s have unkown status"%id_ls[i])
        elif lists[0] == "days_to_death":
            surv_days_ls = lists[1:]
        elif lists[0] == "days_to_last_followup":
            live_days_ls = lists[1:]
        elif lists[0] == "clinical_stage":
            surv_stage_ls = lists[1:]
            # categorical encoding
            # get all stages
            nume_stage_ls = []
            all_stage = set(surv_stage_ls)
            stage_dict = dict(zip(all_stage, list(range(len(all_stage)))))
            with open(join(msk_dir, "stage_dict.json"), 'w') as out_f:
                json.dump(stage_dict, out_f)
            for j in range(len(surv_stage_ls)):
                nume_stage_ls.append(stage_dict[surv_stage_ls[j]])
        elif lists[0] == "days_to_birth":
            #make days positive
            surv_age_ls = lists[1:]
            for k in range(len(surv_age_ls)):
                if surv_age_ls[k] != 'NA':
                    #make it positive
                    surv_age_ls[k] = - int(surv_age_ls[k])
    # integrate survival data
    all_surv_ls = []
    for s in range(len(surv_days_ls)):
        if surv_stat_ls[s] == "alive":
            all_surv_ls.append(live_days_ls[s])
        elif surv_stat_ls[s] == "dead":
            all_surv_ls.append(surv_days_ls[s])
        else:
            all_surv_ls.append("NA")
        
    # convert to dataframe
    print(len(id_ls), len(nume_stat_ls), len(all_surv_ls), len(surv_age_ls), len(nume_stage_ls))
    df_all = pd.DataFrame({'id': id_ls, 
        'survival_status': nume_stat_ls, 
        'survival_days': all_surv_ls,
        'age_days': surv_age_ls,
        'clinical_stage': nume_stage_ls})

    df_all.to_csv(join(msk_dir, out_csv),sep='\t',index=None)


if __name__ == "__main__":
    msk_dir = "/Users/yuexichen/Downloads/lrgr_file/mskfiles"
    id_dir = "/Users/yuexichen/Downloads/lrgr_file/mskfiles/mix_downsize"
    all_clinical = join(msk_dir, "raw-TCGA-OV-all-clinical.tsv")
    out_csv = join(msk_dir,"raw-TCGA-OV-survial-analysis.tsv")
    npyf_dir ="/Users/yuexichen/Downloads/lrgr_file/mskfiles/mix_downsize"
    #row2df(msk_dir, all_clinical, out_csv)
    #id_type = ["WXS", "MSK", "MSK-MSK"]
    #expo_type = ["exposures", "assignments"]
    #ds_list = [("-OV-ds2","-TCGA-OV"), ("-OV-ds2","-OV-ds2"),("-OV-ds5","-TCGA-OV"), ("-OV-ds5","-OV-ds5"),("-OV-ds10","-TCGA-OV"), ("-OV-ds10","-OV-ds10")]
    #expo_type = ["assignments"]
    expo_type = ["expected_topics"]
    ds_list = [("-OV-ds2","-TCGA-OV"), ("-OV-ds2","-OV-ds2")]
    
    all_df_f = out_csv
    for dl in ds_list:
        for et in expo_type:
            id_list_f = join(id_dir, "ov-downsize%s_sample_id.txt"%(re.findall(r'\d+', dl[0])[0]))
            npy_f = join(npyf_dir, "%s%s%s.npy"%(et,dl[0],dl[1]))
            #selected df
            downsized_df_f = join(msk_dir, "downsize-%s%s%s.csv"%(et,dl[0],dl[1]))
            select_df(msk_dir, all_df_f, id_list_f, downsized_df_f)
            new_df_f = join(msk_dir, "heldout-new%s%s-survival-analysis-%s.tsv"%(dl[0],dl[1],et))
            concat_expo(downsized_df_f, npy_f, new_df_f, et)
