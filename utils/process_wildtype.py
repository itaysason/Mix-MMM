import numpy as np
import pandas as pd
from os.path import join
import json

def wilddict(brca_status_f, ov_brca_j, part_type):
    """
    input: file about BRCA status
    output: a json dictionary of ids and wildtype or mutated (all kinds of mutations)
    """
    brca_df = pd.read_csv(brca_status_f, sep='\t')
    group_ls = brca_df['group']
    bina_group = []
    if part_type == 1:
        for gl in group_ls:
              bina_group.append(0)
    elif part_type == 2:
        for gl in group_ls:
            if (gl == "bi_patho"):
                bina_group.append(1)
            else:
                bina_group.append(0)
    elif part_type == 3:
        for gl in group_ls:
            if (gl == "bi_patho") or (gl=="bi_vus"):
                bina_group.append(1)
            else:
                bina_group.append(0)
    elif part_type == 4:
        for gl in group_ls:
            if (gl == "bi_patho") or (gl == "mono_patho"):
                bina_group.append(1)
            else:
                bina_group.append(0)

    brca_dict = dict(zip(list(brca_df['ids']), bina_group))
    with open(ov_brca_j,'w') as out_j:
        json.dump(brca_dict, out_j)

def sele_df(previous_surv_f, wildtype_surv_f, ov_brca_j, platinum_j):
    """
    input: final survival df
    output: selected rows only when it's wildtype and have been treated with platinum
    """
    prev_df = pd.read_csv(previous_surv_f, sep='\t')
    with open(ov_brca_j) as in_f:
        ov_dict = json.load(in_f)
    ov_keys = list(ov_dict.keys())

    with open(platinum_j) as in_f2:
        pt_dict = json.load(in_f2)
    platinum_keys = list(pt_dict.keys())
    # only keep those rows that the last 4 digits of ids match
    id_ls = list(prev_df['id'])
    wild_ls = []
    for i in range(len(prev_df.index)):
        id_4 = id_ls[i][-4:]
        if (id_4 in ov_keys) and (ov_dict[id_4] == 0) and (id_ls[i] in platinum_keys):
                wild_ls.append(prev_df.iloc[i:i+1])
    wild_df = pd.concat(wild_ls)
    wild_df.to_csv(wildtype_surv_f, sep='\t',index=None)

if __name__ == "__main__":
    msk_dir = "/Users/yuexichen/Downloads/lrgr_file/mskfiles"
    brca_status_f = join(msk_dir, "Ovary_OS_KM_matrix.txt")
    ov_brca_j = join(msk_dir, "ov-brca-status.json")
    platinum_j = join(msk_dir, "platinum-dict.json")
    # exp type: MSK-MSK, WXS, MSK
    exp_type = ["WXS", "MSK", "MSK-MSK"]
    #exp_type = ["WXS"]
    # data type: assignments, exposures
    dat_type = ["assignments", "exposures"]
    #dat_type = ["exposures"]
    part_type = [1,2,3,4]
    #dat_type = ["exposures"]
    for pt in part_type:
        for et in exp_type:
            for dt in dat_type:
                previous_surv_f = join(msk_dir, "sig3-final-TCGA-OV-%s-survial-analysis-%s.tsv"%(et, dt))
                wildtype_surv_f = join(msk_dir, "sig3-part%d-BRCA-wildtype-platinum-TCGA-OV-%s-survival-analysis-%s.tsv"%(pt, et, dt))
                wilddict(brca_status_f, ov_brca_j, pt)
                sele_df(previous_surv_f, wildtype_surv_f, ov_brca_j, platinum_j)
