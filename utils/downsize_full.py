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
    mut_id = list(full_df['tumor'])
    #downsize this matrix
    ds_mut = np.zeros(np.shape(mut_val))
    random.seed(seed)
    for i in range(np.shape(mut_val)[0]):
        for j in range(np.shape(mut_val)[1]):
            for k in range(int(mut_val[i][j])): 
                now = random.uniform(0,1)
                if now < 1.0/dsize:
                    ds_mut[i][j] += 1
    # statistics
    print("we have %d mutations per sample"%(ds_mut.sum()/np.shape(ds_mut)[0]))
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
    mix_dir = join(msk_dir, "mixfiles")
    #cancer type: ov or brca
    cancer_type = "brca"
    if cancer_type == "ov":
        dsize = 10
        full_f = join(msk_dir, "sigma-wxs-ov-sbs.tsv")
        sigma_f = join(msk_dir, "sigma-direct-ov-d10.tsv")
        mix_np = join(mix_dir, "wxs-ov-d10_counts.npy") 
        mix_id = join(mix_dir, "wxs-ov-d10_sample_id.txt")
    elif cancer_type == "brca":
        dsize = 500
        full_f = join(msk_dir, "sigma-wgs-brca-sbs.tsv")
        sigma_f = join(msk_dir, "sigma-direct-brca-d500.tsv")
        mix_np = join(mix_dir, "wgs-brca-d500_counts.npy") 
        mix_id = join(mix_dir, "wxs-brca-d500_sample_id.txt")
    downsize(full_f, sigma_f, mix_np, mix_id, dsize)
    

    
