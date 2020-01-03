import pandas as pd
import numpy as np
from os.path import join
import re
#from calc_expdiff import get_index

def get_index(fl_id, ds_id):
    """
    input: full id, downsized id
    output: all position of the downsized id
    """
    #flatten if nested:
    if len(fl_id[0])==1:
        fl_id = [item for sublist in fl_id for item in sublist]
        ds_id = [item for sublist in ds_id for item in sublist]
    fl_dict = dict(zip(fl_id, list(range(len(fl_id)))))
    #print(fl_dict)
    ds_index = []
    for di in ds_id:
        ds_index.append(fl_dict[di])
    return ds_index

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

def sub_max(raw_sbs, recon_M, index_list=None, sub_mat=None):
    """
    index_list: shuffled indices
    sub_mat: the matrix to subtract before normalization
    """
    #raw_sbs = pd.read_csv(raw_M,sep=',')
    # remove the last tumor id column
    raw_sbs_M = np.stack([item[:96] for item in raw_sbs.values])
    # normalize
    if len(index_list)>0:
        #select those rows
        raw_sbs_M = raw_sbs_M[index_list,:]
    #if sub_mat is not None:
        #smooth
    #    raw_sbs_M = raw_sbs_M - sub_mat
    row,col = np.shape(raw_sbs_M)
    for i in range(row):
        for j in range(col):
            if raw_sbs_M[i][j] < sub_mat[i][j]:
                print("position", i,j)
                print(raw_sbs_M[i][j])
                print(sub_mat[i][j])
    sbs_M = raw_sbs_M / raw_sbs_M.sum(axis=1)[:, np.newaxis]
    #print(sbs_M)
    l2_norm = np.linalg.norm(np.matrix(recon_M - sbs_M, dtype=float), ord=2)
    #l2_norm = 0
    return l2_norm

def comp_re_sigma(raw_M, sigma_out, index_list, cosmic):
    """
    compute reconstruction error
    input: the SigMA output file
            input sbs file
            all cosmic signature file
    output: L1 error of M - P*E
    """
    sig_df = pd.read_csv(sigma_out, sep=',')
    #print(list(sig_df))
    exps_all = list(sig_df['exps_all'])
    sigs_all = list(sig_df['sigs_all'])
    #extract numbers in sigs_all
    sig_num = []
    for items in sigs_all:
        sig_num.append(re.findall(r'\d+', items))
    #print(sig_num)
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
    exp_3 = [0]*len(sig_num)
    for i in range(len(sig_num)):
        # for each sample
        recon_row = [0]*96
        for j in range(len(sig_num[i])):
            tmp_sig = cosmic_df.loc[cosmic_df['Signature'] == int(sig_num[i][j])]
            #the first one is the number of signatures
            now_sig = tmp_sig.values[0][1:]
            now_expo = float_exp[i][j]
            recon_row += now_expo * now_sig
            if int(sig_num[i][j]) == 3:
                exp_3[i] = float_exp[i][j]
        row_ls.append(recon_row)
    pererr = sub_max(raw_M, np.stack(row_ls), index_list)
    #print(pererr)
    return pererr, exp_3

def comp_re_ours(raw_M, expo_np, index_list, sig_list, cosmic_f, sub_mat):
    """
    input: 
        raw M: raw sbs, sigma format
        expo_np: numpy exposure file
        index_list: shuffled index list
        sig_list: selected list of signatures
        cosmic_f: the cosmic file
    output: 
    l2 norm 
    """
    cosmic_df = pd.read_csv(cosmic_f, sep='\t')
    row_ls = []
    #all_np =  np.load(expo_np, allow_pickle=True)
    expo_np = expo_np / expo_np.sum(axis=1)[:, np.newaxis]

    for i in range(np.shape(expo_np)[0]):
        recon_row = [0]*96
        for j in range(len(sig_list)):
            tmp_sig = cosmic_df.loc[cosmic_df['Signature'] == int(sig_list[j])]
            #the first one is the number of signatures
            now_sig = tmp_sig.values[0][1:]
            now_expo = expo_np[i][j]
            recon_row += now_expo * now_sig
        row_ls.append(recon_row)
    recon_M = np.stack(row_ls)
    l2norm = sub_max(raw_M, recon_M,index_list, sub_mat)
    return l2norm

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
    """
    print("Now is sigma")
    cancer_type ="ov"
    if cancer_type == "brca":
        #ds_list = ["","-d10","-d100"]
        ds_list = ['all','downsize100','downsize250','downsize500']
    elif cancer_type == "ov":
        #ds_list = ["", "-msk"]
        ds_list = ['all','downsize2','downsize5','downsize10']

    for dl in ds_list:
        if cancer_type == "brca":
            #out_sbs = join(msk_dir, "sigma-wgs-brca%s-sbs.tsv"%dl)
            out_sbs = join(msk_dir, "brca-%s_sigma_sbs.tsv"%dl)
            #sigma_out = join(msk_dir, "sigma-wxs-ov-msk-sbs-out.tsv")
            sigma_out = join(msk_dir,"out-brca-%s_sigma_sbs.tsv"%dl)
            # sigma_formatter(in_sbs, out_sbs)
        elif cancer_type == "ov":
            #out_sbs = join(msk_dir, "sigma-wxs-ov%s-sbs.tsv"%dl)
            out_sbs = join(msk_dir, "ov-%s_sigma_sbs.tsv"%dl)
            #sigma_out = join(msk_dir, "t0-sigma-wxs-ov%s-sbs-out.tsv"%dl)
            sigma_out = join(msk_dir,"out-ov-%s_sigma_sbs.tsv"%dl)
        raw_M = out_sbs
        cutoff = 0
        print("Cancer type: %s"%cancer_type, dl)
        comp_re_sigma(raw_M, sigma_out, out_sbs, cosmic, cutoff,upper=None)
    

    """
    print("Now is MIX")
    ov_sigs = [1, 3, 5]
    brca_sigs = [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]
    # exposures or assignments
    #setting = ["exposures","assignments"]
    setting = ["assignments"]
    # cancer type: ov or brca
    cancer_type = "ov"
    cutoff = 0
    if cancer_type == "brca":
        sig_list = brca_sigs
        ds_list = [("-ICGC-BRCA", "-ICGC-BRCA"), ("-BRCA-ds100","-ICGC-BRCA"),("-BRCA-ds100","-BRCA-ds100"),("-BRCA-ds250","-ICGC-BRCA"),("-BRCA-ds250","-BRCA-ds250"),
        ("-BRCA-ds500","-ICGC-BRCA"),("-BRCA-ds500","-BRCA-ds500")]
    elif cancer_type == "ov":
        sig_list = ov_sigs
        #ds_list = [("",""), ("-msk-region",""), ("-msk-region", "-msk-region")]
        ds_list = [("-TCGA-OV", "-TCGA-OV"), ("-OV-ds2","-TCGA-OV"), ("-OV-ds2","-OV-ds2"),("-OV-ds5","-TCGA-OV"), ("-OV-ds5","-OV-ds5"),("-OV-ds10","-TCGA-OV"), ("-OV-ds10","-OV-ds10")]
    expo_dir = "/Users/yuexichen/Downloads/lrgr_file/mskfiles/mix_downsize"
    ratio = []
    for st in setting:
        for dl in ds_list:
            print("Now the setting is %s"%st)
            print(dl)
            expo_np = join(expo_dir,"%s%s%s.npy"%(st, dl[0],dl[1]))
            re_int =  re.findall(r'\d+', dl[0])
            if re_int:
                ratio = int(re_int[0])
                ds_id = pd.read_csv(join(msk_dir, "mix_downsize/%s-downsize%d_sample_id.txt"%(cancer_type,ratio)),sep='\n', header=None).values.tolist()
            else:
                ds_id = pd.read_csv(join(msk_dir, "mix_downsize/%s-original_sample_id.txt"%(cancer_type)),sep='\n', header=None).values.tolist()

            raw_M = join(msk_dir, "%s-all_sigma_sbs.tsv"%cancer_type)
            fl_id = pd.read_csv(join(msk_dir, "mix_downsize/%s-original_sample_id.txt"%cancer_type), sep='\n', header=None).values.tolist()
            index_list = get_index(fl_id, ds_id)
            #print(index_list)
            pererr, exp_3 = comp_re_ours(raw_M, expo_np, index_list, sig_list, cosmic, st, cutoff)
            np.save(join(msk_dir, "mix_downsize/sigma_downsize%d_sig3.npy"%dl), exp_3, allow_pickle=True)