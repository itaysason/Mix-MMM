import numpy as np
import pandas as pd
from os.path import join

def sig_plus(surv_f, sig_exp_f, inner_f):
    # get survival file, inner join with sigma exposure file
    surv_df = pd.read_csv(surv_f, sep = '\t')
    sig_exp_df = pd.read_csv(sig_exp_f, sep = '\t')
    merged_inner = pd.merge(left=surv_df,right=sig_exp_df, left_on='id', right_on='id')
    merged_inner.to_csv(inner_f, index=None, sep='\t')

def sigma_col(sigma_f, sig_exp_f):
    # get a dictionary from sigma
    # add new columns
    # truncate id at sample level: e.g. TCGA-04-1331
    sigma_df = pd.read_csv(sigma_f, sep=',')
    # sigma_f['tumor']
    # sigs_all,exps_all
    # sig3_dict = dict(zip(sigma_f['tumor'], sigma_f['']))
    id_ls = [item[:12] for item in list(sigma_df['tumor'])]
    sig3_bina = []
    sig3_exposure = []
    sig_ls = list(sigma_df['sigs_all'])
    exp_ls = list(sigma_df['exps_all'])
    for s in range(len(exp_ls)):
        now_sig = sig_ls[s].split('.')
        now_exp = exp_ls[s].split('_')
        now_dict = dict(zip(now_sig, now_exp))
        #print(list(now_dict.keys()))
        if 'Signature_3' in list(now_dict.keys()):
            sig3_bina.append(1)
            sig3_exposure.append(now_dict['Signature_3'])
        else:
            sig3_bina.append(0)
            sig3_exposure.append(0)
    
    tmp_df = pd.DataFrame([id_ls, sig3_bina, sig3_exposure])
    new_df = tmp_df.transpose() 
    new_df.columns=['id','sigma_binary_sig3', 'sigma_exposure_sig3']
    new_df.to_csv(sig_exp_f, index=None, sep='\t')

if __name__ == "__main__":
    # input: sigmat out file
    #       survival file
    # output: sigma-wildtype-TCGA-OV-MSK-MSK-survial-analysis.tsv
    #       two new columns: binary sigma exposures, continous sigma exposures
    msk_dir ="/Users/yuexichen/Downloads/lrgr_file/mskfiles"
    # type: wxs or msk
    data_type = "msk"
    sigma_f = join(msk_dir, "sigma-wxs-ov-%s-sbs-out.tsv"%data_type)
    sig_exp_f = join(msk_dir, "sigma-%s-sig3-bina-exposure.csv"%data_type)
    sigma_col(sigma_f, sig_exp_f)
    surv_f = join(msk_dir, "final-TCGA-OV-%s-survial-analysis-exposures.tsv"%data_type.upper())
    inner_f = join(msk_dir, "sigma-TCGA-OV-%s-survival-analysis-exposures.tsv"%data_type.upper())
    sig_plus(surv_f, sig_exp_f, inner_f)
