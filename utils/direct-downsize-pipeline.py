import numpy as np
import pandas as pd
from os.path import join
import random

def downsize(full_f, sigma_f, mix_np, mix_id, dsize, seed=1234):
    """
    input: full sigma, a parameter for downsize
    output: downsized data in sigma format
    """
    full_df = pd.read_csv(full_f, sep=',')
    mut_val = full_df.values[:,:-1]
    tmp_mut_id = list(full_df['tumor'])
    #downsize this matrix
    tmp_ds_mut = np.zeros(np.shape(mut_val))
    random.seed(seed)
    for i in range(np.shape(mut_val)[0]):
        for j in range(np.shape(mut_val)[1]):
            for k in range(int(mut_val[i][j])): 
                now = random.uniform(0,1)
                if now < 1.0/dsize:
                    tmp_ds_mut[i][j] += 1
    # statistics
    print("we have %d mutations per sample"%(tmp_ds_mut.sum()/np.shape(tmp_ds_mut)[0]))

    #ignore rowsum equals to 0
    mut_id = []
    rowsum = list(tmp_ds_mut.sum(axis=1))
    sele = []
    for m in range(len(tmp_mut_id)):
        if rowsum[m] > 0:
            mut_id.append(tmp_mut_id[m])
            sele.append(m)
    ds_mut = tmp_ds_mut[sele] 
    #save to numpy file
    np.save(mix_np, ds_mut)
    #save to id file
    with open(mix_id,'w') as mf:
        for mi in mut_id:
            mf.write(mi + '\n')
    #save to sigma
    sigma_df = pd.DataFrame(data=ds_mut, columns=list(full_df)[:-1])
    sigma_df['tumor'] = mut_id
    sigma_df.to_csv(sigma_f, sep=',', index=None)

if __name__ == "__main__":
    """
    input: wxs/wgs
    output: downsized 10times directly from WGS or WXS, npy files
    """
    msk_dir = "/Users/yuexichen/Downloads/lrgr_file/mskfiles"
    #cancer type: ov or brca
    cancer_type = "brca"
    if cancer_type == "ov":
        dsize = [2, 5, 10]
        for dz in dsize:
            full_f = join(msk_dir, "ov-all_sigma_sbs.tsv")
            mix_dir = join(msk_dir, "ov_downsize")
            sigma_f = join(msk_dir, "ov-donwsize%d_sigma_sbs.tsv"%dz)
            mix_np = join(mix_dir, "ov-downsize%d_counts.npy"%dz) 
            mix_id = join(mix_dir, "ov-downsize%d_sample_id.txt"%dz)
            downsize(full_f, sigma_f, mix_np, mix_id, dz)
    elif cancer_type == "brca":
        dsize = [100,250,500]
        for dz in dsize:
            full_f = join(msk_dir, "brca-all_sigma_sbs.tsv")
            mix_dir = join(msk_dir, "brca_downsize")
            sigma_f = join(msk_dir, "brca-downsize%d_sigma_sbs.tsv"%dz)
            mix_np = join(mix_dir, "brca-downsize%d_counts.npy"%dz) 
            mix_id = join(mix_dir, "brca-downsize%d_sample_id.txt"%dz)
            downsize(full_f, sigma_f, mix_np, mix_id, dz)
    

    
