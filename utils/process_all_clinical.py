from os.path import join
import json
import pandas as pd
import numpy as np

def concat_expo(select_df_f, npy_f, new_df_f):
    """
    input:
    1. the selected dataframe
    2. the numpy file of exposures
    output:
    A new dataframe with the new columns
    column 1: binary exposures of signature 3
    column 2: continous exposures of signature 3
    """
    select_df = pd.read_csv(select_df_f, sep='\t')
    all_npy = np.load(npy_f, allow_pickle=True)
    expo_3 = [items[1] for items in all_npy]
    #only take the second element, i.e. the exposure to signature 3
    bina_3 = []
    zero_cnt = 0
    for expo in expo_3:
        if expo == 0:
            bina_3.append(0)
            zero_cnt += 1
        else:
            bina_3.append(1)
    print("We have %d zero counts"%zero_cnt)
    select_df['binary_sig3'] = bina_3
    select_df['exposure_sig3'] = expo_3
    select_df.to_csv(new_df_f, sep='\t', index=None)


def select_df(msk_dir, all_df_f, id_list_f, select_df_f):
    """
    input: 
    1. the whole table of survival dataframe
    2. the patients id list
    output:
    dataframes of rows selected by this id list
    """
    all_df = pd.read_csv(all_df_f, sep='\t')
    with open(id_list_f) as in_f:
        id_list = [item.rstrip() for item in in_f.readlines()]
    #enforce the order
    row_ls = []
    for il in id_list:
        now_row = all_df.loc[all_df['id'] == il]
        row_ls.append(now_row)

    select_df = pd.concat(row_ls)
    select_df.to_csv(select_df_f, sep='\t', index=None)


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
    all_clinical = join(msk_dir, "raw-TCGA-OV-all-clinical.tsv")
    out_csv = join(msk_dir,"raw-TCGA-OV-survial-analysis.tsv")
    npyf_dir ="/Users/yuexichen/Downloads/tcga-ov-exposures"
    row2df(msk_dir, all_clinical, out_csv)
    id_type = "MSK-MSK"
    #id_type = "WXS"
    all_df_f = out_csv
    if id_type == "MSK":
        id_list_f = "/Users/yuexichen/Desktop/LRGR/Repository/Mix-MMM/data/WXS-TCGA-OV/wxs-ov-msk-region_sample_id.txt"
        npy_f = join(npyf_dir, "exposures-TCGA-OV-msk-region-TCGA-OV.npy")
    elif id_type == "WXS":
        id_list_f = "/Users/yuexichen/Desktop/LRGR/Repository/Mix-MMM/data/WXS-TCGA-OV/wxs-ov-all_sample_id.txt"
        npy_f = join(npyf_dir, "exposures-TCGA-OV-TCGA-OV.npy")
    elif id_type == "MSK-MSK":
        id_list_f = "/Users/yuexichen/Desktop/LRGR/Repository/Mix-MMM/data/WXS-TCGA-OV/wxs-ov-msk-region_sample_id.txt"
        npy_f = join(npyf_dir, "exposures-TCGA-OV-msk-region-TCGA-OV-msk-region.npy")
    select_df_f = join(msk_dir, "TCGA-OV-select-%s.csv"%id_type)
    select_df(msk_dir, all_df_f, id_list_f, select_df_f)
    new_df_f = join(msk_dir, "final-TCGA-OV-%s-survial-analysis.tsv"%id_type)
    concat_expo(select_df_f, npy_f, new_df_f)
