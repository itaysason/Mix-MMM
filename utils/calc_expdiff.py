import numpy as np
import pandas as pd
import re
from os.path import join

def expo_diff(full_expo, ds_expo):
    """
    input: exposures on full data and downsized data separately
    output: L2 norm
    """
    l2_norm = np.linalg.norm(np.matrix(full_expo - ds_expo, dtype=float), ord=2)
    return l2_norm

def get_sigma_expo(raw_sigmaout):
    """
    input: raw output file of sigma
    output: normalized exposures
    """
    sig_df = pd.read_csv(raw_sigmaout, sep=',')
    #print(list(sig_df))
    exps_all = list(sig_df['exps_all'])
    sigs_all = list(sig_df['sigs_all'])
    id_all = list(sig_df['tumor'])
    #extract numbers in sigs_all
    sig_num = []
    sig_set = []
    for items in sigs_all:
        sig_num.append(re.findall(r'\d+', items))
        sig_set.extend(re.findall(r'\d+', items))
    sig_set = list(set(sig_set))
    sig_dict = dict(zip(sig_set, list(range(len(sig_set)))))
    #normalize exposures
    float_exp = []
    for fe in exps_all:
        tmp = fe.split('_')
        tmp = [float(i) for i in tmp]
        if sum(tmp) > 0:
            tmp = [i/sum(tmp) for i in tmp]
        float_exp.append(tmp)

    #print(float_exp)
    #put exposures back
    sigma_ep = np.zeros([len(sig_num), len(sig_set)])
    for i in range(len(sig_num)):
        for j in range(len(sig_num[i])):
            #in case the signature is not ordered
            sigma_ep[i][sig_dict[sig_num[i][j]]] = float_exp[i][j] 
    #print(sigma_ep)

    return sigma_ep, id_all

def get_mix_expo(mix_npy):
    """
    input: npy file of mix asignments files
    output: normalized exposures
    """
    all_np =  np.load(mix_npy, allow_pickle=True)
    all_np = all_np / all_np.sum(axis=1)[:, np.newaxis]
    return all_np

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

if __name__ == "__main__":
    #ov_sigs = [1, 3, 5]
    #brca_sigs = [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]
    msk_dir = "/Users/yuexichen/Downloads/lrgr_file/mskfiles"
    cancer_type = "brca"
    method ="mix"
    if cancer_type == "ov":
        ds_ls = [2, 5, 10]
    elif cancer_type == "brca":
        ds_ls = [100, 250, 500]
    if method == "sigma":
        for dl in ds_ls:
            full_sigma = join(msk_dir,"out-%s-all_sigma_sbs.tsv"%cancer_type)
            ds_sigma = join(msk_dir,"out-%s-downsize%d_sigma_sbs.tsv"%(cancer_type, dl))
            fl_sigma_expo, fl_id = get_sigma_expo(full_sigma)
            ds_sigma_expo, ds_id = get_sigma_expo(ds_sigma)
            ds_index = get_index(fl_id, ds_id)
            sele_fl_sigma_expo = fl_sigma_expo[ds_index,:]
            l2norm = expo_diff(sele_fl_sigma_expo, ds_sigma_expo)
            print("When the cancer type = %s, method=%s, downsize ratio=%s, the l2norm difference is %0.4f"%(cancer_type, method, dl, l2norm))
    elif method == "mix":
        for dl in ds_ls:
            if cancer_type == "brca":
                full_mix = join(msk_dir, 'mix_downsize/assignments-ICGC-BRCA-ICGC-BRCA.npy')
                #ds_mix = join(msk_dir,'mix_downsize/assignments-BRCA-ds%d-BRCA-ds%d.npy'%(dl, dl))
                ds_mix = join(msk_dir,'mix_downsize/assignments-BRCA-ds%d-ICGC-BRCA.npy'%(dl))
                #ds_mix = join(msk_dir,'mix_downsize/assignments-BRCA-ds%d-BRCA-ds%d.npy'%(dl, dl))
            elif cancer_type == "ov":
                full_mix = join(msk_dir, 'mix_downsize/assignments-TCGA-OV-TCGA-OV.npy')
                ds_mix = join(msk_dir, 'mix_downsize/assignments-OV-ds%d-TCGA-OV.npy'%dl)
                #ds_mix = join(msk_dir, 'mix_downsize/assignments-OV-ds%d-OV-ds%d.npy'%(dl, dl))

            fl_mix_expo = get_mix_expo(full_mix)
            ds_mix_expo = get_mix_expo(ds_mix)
            fl_id = pd.read_csv(join(msk_dir, "mix_downsize/%s-original_sample_id.txt"%cancer_type), sep='\n', header=None).values.tolist()
            ds_id = pd.read_csv(join(msk_dir, "mix_downsize/%s-downsize%d_sample_id.txt"%(cancer_type,dl)),sep='\n', header=None).values.tolist()
            ds_index = get_index(fl_id, ds_id)
            #print(np.shape(fl_mix_expo))
            sele_fl_mix_expo = fl_mix_expo[ds_index,:]
            #print(np.shape(sele_fl_mix_expo))
            l2norm=expo_diff(sele_fl_mix_expo, ds_mix_expo)
            print("When the cancer type = %s, method=%s, downsize ratio=%s, the l2norm difference is %0.4f"%(cancer_type, method, dl, l2norm))
    
