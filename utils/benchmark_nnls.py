import numpy as np
import pandas as pd
from os.path import join
from scipy.optimize import nnls

if __name__ == "__main__":
    msk_dir = "/Users/yuexichen/Downloads/lrgr_file/mskfiles"
    cosmic = join(msk_dir, "cosmic-signatures.tsv")
    cosmic_df = pd.read_csv(cosmic, sep='\t')
    sig_ls = [1,3,5]
    sig_mat = []
    for j in range(len(sig_ls)):
        tmp_sig = cosmic_df.loc[cosmic_df['Signature'] == int(sig_ls[j])]
        #the first one is the number of signatures
        now_sig = tmp_sig.values[0][1:]
        sig_mat.append(now_sig)
    sig_mat = np.transpose(np.array(sig_mat))
    #print(np.shape(sig_mat))
    ds_ls = [2,5,10]
    part_ls = [1,2]
    for p in part_ls:
        for i in ds_ls:
            expo_mat = []
            M_f = join(msk_dir, "mix_downsize/models/OV-ds%d-part%d-data.npy"%(i, p))
            mut_b = np.transpose(np.load(M_f))
            for col in range(np.shape(mut_b)[1]):
                expo,_ = nnls(sig_mat, mut_b[:,col])
                expo_mat.append(expo)
            print(np.shape(expo_mat))
            np.save(join(msk_dir, "mix_downsize/models/nnls-OV-ds%d-part%d-exposure.npy"%(i, p)),expo_mat, allow_pickle=True)