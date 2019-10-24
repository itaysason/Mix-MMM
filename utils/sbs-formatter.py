import pandas as pd
import numpy as np
from os.path import join
import re

def sigma_formatter(in_sbs, out_sbs):
    """
    turn our SBS format to SigMA format
    input: our SBS format
    output: SigMA SBS format
    """
    in_df = pd.read_csv(in_sbs, sep='\t')
    #change the header line
    old_name =in_df.columns.values.tolist()
    new_name = ["tumor"]
    #ignore the first unnamed column
    for item in old_name[1:]:
        new_name.append(item[2]+item[4]+item[0]+item[-1])
    # convert to lower letter
    new_name = [letter.lower() for letter in new_name]
    in_df.columns = new_name
    #change position of columns
    cols = in_df.columns.tolist()
    #switch order
    cols = cols[1:] + [cols[0]]
    in_df = in_df[cols]
    in_df.to_csv(out_sbs, sep=',', index=None)

def comp_re(sigma_out, out_sbs, cosmic):
    """
    compute reconstruction error
    input: the SigMA output file
            input sbs file
            all cosmic signature file
    output: L1 error of M - P*E
    """
    sig_df = pd.read_csv(sigma_out, sep=',')
    #print(list(sig_df))
    sigs_all = list(sig_df['sigs_all'])
    exps_all = list(sig_df['exps_all'])
    #extract numbers in sigs_all
    sig_num = []
    for items in sigs_all:
        sig_num.append(re.findall(r'\d+', items))
    #normalize exposures
    float_exp = []
    for fe in exps_all:
        tmp = fe.split('_')
        tmp = [float(i) for i in tmp]
        tmp = [i/sum(tmp) for i in tmp]
        float_exp.append(tmp)
    #select signatures
    cosmic_df = pd.read_csv(cosmic, sep='\t')
    row_ls = []
    for i in range(len(sig_num)):
        # for each sample
        recon_row = [0]*96
        for j in range(len(sig_num[i])):
            tmp_sig = cosmic_df.loc[cosmic_df['Signature'] == int(sig_num[i][j])]
            #the first one is the number of signatures
            now_sig = tmp_sig.values[0][1:]
            now_expo = float_exp[i][j]
            recon_row += now_expo * now_sig
        row_ls.append(recon_row)
    
    raw_sbs = pd.read_csv(out_sbs,sep=',')
    #remove the last column
    all_sbs_M = np.stack([item[:96] for item in raw_sbs.values])
    # remove rows that have sum smaller than 5
    sbs_M_ls = []
    for i in range(np.shape(all_sbs_M)[0]):
        #the threshold is 10 for WGS/WXS
        if sum(all_sbs_M[i,:])>=5:
            sbs_M_ls.append(all_sbs_M[i,:])    
    sbs_M = np.stack(sbs_M_ls)
    recon_M = np.stack(row_ls)
    #error per mutation
    per_err = (abs(recon_M-sbs_M)).sum()/sbs_M.sum()
    print(per_err)
    

if __name__ == "__main__":
    #input: our SBS format
    #output: SigMA SBS format
    #difference: the header line names, the sep, and the position of tumor id
    msk_dir = "/Users/yuexichen/Downloads/lrgr_file/mskfiles"
    cosmic = join(msk_dir, "cosmic-signatures.tsv")
    # original WGS
    # in_sbs = join(msk_dir, "wgs-brca-sbs.tsv")
    # out_sbs = join(msk_dir, "sigma-wgs-brca-sbs.tsv")
    # sigma_out = join(msk_dir, "sigma-wgs-brca-sbs-out.tsv")

    # downsize 10% WGS
    # in_sbs = join(msk_dir, "wgs-brca-d10-sbs.tsv")
    # out_sbs = join(msk_dir, "sigma-wgs-brca-d10-sbs.tsv")
    # sigma_out = join(msk_dir, "sigma-wgs-brca-d10-sbs-out.tsv")

    # downsize 1% WGS
    # in_sbs = join(msk_dir, "wgs-brca-d100-sbs.tsv")
    # out_sbs = join(msk_dir, "sigma-wgs-brca-d100-sbs.tsv")
    # sigma_out = join(msk_dir, "sigma-wgs-brca-d100-sbs-out.tsv")
    
    # TCGA WXS
    # in_sbs = join(msk_dir, "wxs-ov-sbs.tsv")
    # out_sbs = join(msk_dir, "sigma-wxs-ov-sbs.tsv")
    # sigma_out = join(msk_dir, "sigma-wxs-ov-sbs-out.tsv")

    # TCGA MSK
    in_sbs = join(msk_dir, "wxs-ov-msk-sbs.tsv")
    out_sbs = join(msk_dir, "sigma-wxs-ov-msk-sbs.tsv")
    sigma_out = join(msk_dir, "sigma-wxs-ov-msk-sbs-out.tsv")

    # sigma_formatter(in_sbs, out_sbs)
    comp_re(sigma_out, out_sbs, cosmic)

   

    