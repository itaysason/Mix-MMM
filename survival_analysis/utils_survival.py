import numpy as np
import pandas as pd
import argparse




def main(args):
    comb_df=pd.DataFrame()
    # read brca group
    brca_df=pd.read_csv(args.brca_label, sep='\t')
    # read age
    age_df = pd.read_csv(args.age_label, sep=',')
    #read drug
    with open(args.drug_label) as in_f:
        tmp = [x.strip().split('\t') for x in in_f.read().splitlines()]
    #only select interesting rows
    sele_dict={}
    sele_marker=['patient.bcr_patient_barcode','drug_name', 'patient.stage_event.clinical_stage']
    for rows in tmp:
        for sm in sele_marker:
            if sm in rows[0]:
                sele_dict[rows[0]]=rows[1:]
                #print("For %s, the length is %s"%(rows[0], len(rows[1:])))
    comb_ls=pd.DataFrame(data=sele_dict).values.tolist()
    patient_ls = []
    for row in comb_ls:
        add_flag=False
        for items in row:
            if ('cisplatin' in items) and (add_flag==False): 
                patient_ls.append(row[0])
                add_flag=True
            elif ('carboplatin' in items) and (add_flag==False):
                patient_ls.append(row[0])
                add_flag=True
            

    ids = [str(items) for items in list(brca_df['ids'])]
    age_dict = dict(zip(age_df['PATIENT ID'], age_df['Age']))
    stage_dict = dict(zip(age_df['PATIENT ID'], age_df['stage']))
    surv_dict = dict(zip(ids, brca_df['death']))
    surv_stat_dict = dict(zip(ids, brca_df['status']))
    group_dict = dict(zip(ids, brca_df['group']))
    new_patient_ls = []
    stage_ls = []
    age_ls = []
    surv_ls = []
    stat_ls = []
    group_ls = []
    #print(patient_ls)
    for pl in patient_ls:
        pl_4 = pl[-4:]
        #print(pl_4)
        if pl_4 in surv_dict:
            surv_ls.append(surv_dict[pl_4])
        else:
            continue
        if pl_4 in surv_stat_dict:
            stat_ls.append(surv_stat_dict[pl_4])
        else:
            continue
        if pl_4 in group_dict:
            group_ls.append(group_dict[pl_4])
        else:
            continue
        if pl.upper() in age_dict:
            age_ls.append(age_dict[pl.upper()])
        else:
            continue
        if pl.upper() in stage_dict:
            stage_ls.append(stage_dict[pl.upper()])
        new_patient_ls.append(pl)

    comb_df['id']=new_patient_ls
    comb_df['age']=age_ls
    comb_df['stage']=stage_ls
    comb_df['survival_days'] = surv_ls
    comb_df['survival_status'] = stat_ls
    comb_df['group'] = group_ls
    #print(comb_df)

    comb_df.to_csv(args.skeleton_df, sep='\t', index=None)

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-al', '--age_label')
    parser.add_argument('-dl', '--drug_label')
    parser.add_argument('-bl', '--brca_label')
    parser.add_argument('-sd', '--skeleton_df')
    args = parser.parse_args()    
    main(args)



