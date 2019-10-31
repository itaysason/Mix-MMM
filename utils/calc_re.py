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

def sub_max(raw_M, recon_M, cutoff,upper):
    raw_sbs = pd.read_csv(raw_M,sep=',')
    #remove the last column
    all_sbs_M = np.stack([item[:96] for item in raw_sbs.values])
    #print("raw M", np.shape(all_sbs_M))
    # remove rows that have sum smaller than 5
    sbs_M_ls = []
    sele_row = []
    for i in range(np.shape(all_sbs_M)[0]):
        #the threshold is 10 for WGS/WXS
        if sum(all_sbs_M[i,:])>cutoff:
            sbs_M_ls.append(all_sbs_M[i,:])
            sele_row.append(i)
        #if sum(all_sbs_M[i,:]) < upper:
        #    sele_row.append(i)

    sbs_M = np.stack(sbs_M_ls)
    sum_mut = sbs_M.sum(axis=1).astype(float)
    #print(sum_mut)
    #print("number of samples: ", (np.shape(sum_mut)))
    #print("Mutations in raw count max: , min: , average: , mean: ",np.amax(sum_mut), np.amin(sum_mut), np.average(sum_mut), np.mean(sum_mut)) 
    #error per mutation
    #not for sigma
    #if len(sele_row) > 0:
    #    recon_M = recon_M[sele_row]
    #print("%d rows have been selected in the matrix"%len(sele_row))
    #per_err = (abs(recon_M-sbs_M)).sum()/sbs_M.sum()
    l1_norm = np.linalg.norm((recon_M - sbs_M), ord=1)/(np.linalg.norm(sbs_M, ord=1))
    print(raw_M)
    print("l1 norm %0.3f"%l1_norm)
    #assert(per_err==l1_norm)
    #print(per_err)
    return l1_norm

def comp_re_sigma(raw_M, sigma_out, out_sbs, cosmic, cutoff, upper):
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
        if sum(tmp) > 0:
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
    pererr = sub_max(raw_M, np.stack(row_ls), cutoff, upper)
    #print(pererr)
    return pererr

def comp_re_ours(raw_M, expo_np, sig_list, cosmic, setting, cutoff,upper):
    # input: 
    # raw M: raw sbs file, sigma format
    # expo_np: numpy exposure file
    # sig_list: selected list of signatures
    # cosmic: a list of signatures
    # setting: assignments or exposures
    # output: 
    # errors per mutation 
    # select signatures
    cosmic_df = pd.read_csv(cosmic, sep='\t')
    row_ls = []
    all_np =  np.load(expo_np, allow_pickle=True)
    for i in range(np.shape(all_np)[0]):
        # for each sample
        recon_row = [0]*96
        #print(sig_list)
        for j in range(len(sig_list)):
            tmp_sig = cosmic_df.loc[cosmic_df['Signature'] == int(sig_list[j])]
            #the first one is the number of signatures
            now_sig = tmp_sig.values[0][1:]
            #now_expo = float_exp[j]
            if setting == "assignments":
                if sum(all_np[i])>0:
                    now_expo = all_np[i][j]/sum(all_np[i])
                else:
                    now_expo = all_np[i][j]
            elif setting == "exposures":
                now_expo = all_np[i][j]
            elif setting == "expected_topics":
                now_expo = np.exp(all_np[i][j])/sum(np.exp(all_np[i]))
            recon_row += now_expo * now_sig
        row_ls.append(recon_row)
    recon_M = np.stack(row_ls)
    #print(np.shape(recon_M))
    pererr = sub_max(raw_M, recon_M,cutoff,upper)
    return pererr

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
    #in_sbs = join(msk_dir, "wxs-ov-msk-sbs.tsv")
    # sigma sbs for msk
    #out_sbs = join(msk_dir, "sigma-wxs-ov-msk-sbs.tsv")
    
    # cancer type: ov or brca
    
    
    cancer_type ="ov"
    if cancer_type == "brca":
        #ds_list = ["","-d10","-d100"]
        ds_list = [("-new-BRCA","-ICGC-BRCA"),("-new-BRCA","-new-BRCA")]
    elif cancer_type == "ov":
        #ds_list = ["", "-msk"]
        ds_list = [("-new-OV","-TCGA-OV"), ("-new-OV","-new-OV")]

    for dl in ds_list:
        if cancer_type == "brca":
            #out_sbs = join(msk_dir, "sigma-wgs-brca%s-sbs.tsv"%dl)
            out_sbs = join(msk_dir, "sigma-direct-brca-d500.tsv")
            #sigma_out = join(msk_dir, "sigma-wxs-ov-msk-sbs-out.tsv")
            sigma_out = join(msk_dir,"t0-sigma-direct-brca-d500-out.tsv")
            # sigma_formatter(in_sbs, out_sbs)
        elif cancer_type == "ov":
            #out_sbs = join(msk_dir, "sigma-wxs-ov%s-sbs.tsv"%dl)
            out_sbs = join(msk_dir, "sigma-direct-ov-d10.tsv")
            #sigma_out = join(msk_dir, "t0-sigma-wxs-ov%s-sbs-out.tsv"%dl)
            sigma_out = join(msk_dir,"t0-sigma-direct-ov-d10-out.tsv")
        if len(dl)>0:
                #panels
                upper = 5
        else:
                #wxs or wgs
                upper = 10
        raw_M = out_sbs
        cutoff = 0
        comp_re_sigma(raw_M, sigma_out, out_sbs, cosmic, cutoff,upper)
    

    """
    ov_sigs = [1, 3, 5]
    brca_sigs = [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]
    # exposures or assignments
    #setting = ["exposures","assignments"]
    setting = ["assignments"]
    # cancer type: ov or brca
    cancer_type = "brca"
    cutoff = 0
    if cancer_type == "brca":
        sig_list = brca_sigs
        #ds_list = [("",""),("-ds10",""),("-ds10","-ds10"), ("-ds100",""), ("-ds100","-ds100")]
        ds_list = [("-new-BRCA","-ICGC-BRCA"),("-new-BRCA","-new-BRCA")]
    elif cancer_type == "ov":
        sig_list = ov_sigs
        #ds_list = [("",""), ("-msk-region",""), ("-msk-region", "-msk-region")]
        ds_list = [("-new-OV","-TCGA-OV"), ("-new-OV","-new-OV")]
    expo_dir = "/Users/yuexichen/Downloads/lrgr_file/mskfiles/direct_ds"
    
    for st in setting:
        for dl in ds_list:
            print("Now the setting is %s"%st)
            if cancer_type == "brca":
                #expo_np = join(expo_dir, "%s-ICGC-BRCA%s-ICGC-BRCA%s.npy"%(st,dl[0],dl[1]))
                expo_np = join(expo_dir,"%s%s%s.npy"%(st, dl[0],dl[1]))
                #raw_M = join(msk_dir, "sigma-wgs-%s%s-sbs.tsv"%(cancer_type,dl[0]))
                raw_M = join(msk_dir, "sigma-direct-brca-d500.tsv")
            elif cancer_type == "ov":
                #expo_np = join(expo_dir, "%s-TCGA-OV%s-TCGA-OV%s.npy"%(st, dl[0],dl[1]))
                expo_np = join(expo_dir,"%s%s%s.npy"%(st, dl[0],dl[1]))
                #raw_M = join(msk_dir, "sigma-wxs-%s%s-sbs.tsv"%(cancer_type,dl[0]))
                raw_M = join(msk_dir, "sigma-direct-ov-d10.tsv")
            if len(dl[0])>0:
                #panels
                upper = 5
            else:
                #wxs or wgs
                upper = 10
            pererr = comp_re_ours(raw_M, expo_np, sig_list, cosmic, st, cutoff, upper)
            #print("file name: ", expo_np)
            #print(pererr)
    """