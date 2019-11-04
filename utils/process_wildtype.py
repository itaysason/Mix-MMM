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
            if (gl == "rest"):
                bina_group.append(0)
            else:
                bina_group.append(1)
    brca_dict = dict(zip(list(brca_df['ids']), bina_group))
    with open(ov_brca_j,'w') as out_j:
        json.dump(brca_dict, out_j)

def sele_df(previous_surv_f, doc_surv_pref, ov_brca_j):
    """
    input: final survival df, threshold of being sig3+
    output: selected rows only when its wildtype status is known, and a status column which indicates its sig3 status and BRCAness
    """
    prev_df = pd.read_csv(previous_surv_f, sep='\t')
    with open(ov_brca_j) as in_f:
        ov_dict = json.load(in_f)
    ov_keys = list(ov_dict.keys())

    # only keep those rows that the last 4 digits of ids match
    id_ls = list(prev_df['id'])
    doc_ls = []
    for i in range(len(prev_df.index)):
        id_4 = id_ls[i][-4:]
        if (id_4 in ov_keys):
            doc_ls.append(prev_df.iloc[i:i+1])
    doc_df = pd.concat(doc_ls)
    all_exp = [float(item) for item in list(doc_df['exposure_sig3'])]
    percent = [0,10,25,50]
    for perc in percent:
        status = []
        perc_bina = []
        sig_threshold = sorted(all_exp)[int(len(all_exp)*perc/100.0)]
        for j in range(len(doc_df.index)):
            id_4 = list(doc_df['id'])[j][-4:]
            now_expo = float(list(doc_df['exposure_sig3'])[j])
            if (ov_dict[id_4] == 0) and (now_expo <=sig_threshold):
                now_stat = "Sig3-"
                status.append(now_stat)
                perc_bina.append(0)
            elif (ov_dict[id_4] == 0) and (now_expo > sig_threshold):
                now_stat = "Sig3+"
                status.append(now_stat)
                perc_bina.append(1)
            elif (ov_dict[id_4] == 1):
                now_stat = "BRCA mutations"
                status.append(now_stat)
                perc_bina.append("na")

        doc_df['status'] = status
        doc_df['perc_bina']=perc_bina
        doc_df.to_csv(doc_surv_pref + "-%d.tsv"%perc, sep='\t',index=None)

if __name__ == "__main__":
    msk_dir = "/Users/yuexichen/Downloads/lrgr_file/mskfiles"
    brca_status_f = join(msk_dir, "Ovary_OS_KM_matrix.txt")
    ov_brca_j = join(msk_dir, "ov-brca-status.json")
    #platinum_j = join(msk_dir, "platinum-dict.json")
    # exp type: MSK-MSK, WXS, MSK
    #exp_type = ["WXS", "MSK", "MSK-MSK"]
    #ds_list = [("-OV-ds2","-TCGA-OV"), ("-OV-ds2","-OV-ds2"),("-OV-ds5","-TCGA-OV"), ("-OV-ds5","-OV-ds5"),("-OV-ds10","-TCGA-OV"), ("-OV-ds10","-OV-ds10")]
    ds_list = [("-OV-ds2","-TCGA-OV"), ("-OV-ds2","-OV-ds2")]
    # data type: assignments, exposures
    #dat_type = ["assignments", "exposures"]
    #dat_type = ["assignments"]
    dat_type = ["expected_topics"]
    part_type = [1]
    #dat_type = ["exposures"]
    for pt in part_type:
        for dl in ds_list:
            for dt in dat_type:
                #previous_surv_f = join(msk_dir, "sig3-final-TCGA-OV-%s-survial-analysis-%s.tsv"%(et, dt))
                previous_surv_f = join(msk_dir, "heldout-new%s%s-survival-analysis-%s.tsv"%(dl[0],dl[1],dt))
                doc_surv_pref = join(msk_dir, "part%d-status-new%s%s-survival-analysis-%s"%(pt, dl[0], dl[1], dt))
                wilddict(brca_status_f, ov_brca_j, pt)
                sele_df(previous_surv_f, doc_surv_pref, ov_brca_j)
