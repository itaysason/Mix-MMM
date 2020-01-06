import pandas as pd
import numpy as np
from os.path import join

def npy2sigma_nume(npyf, index_ls, cosmicf,sigmaf):
    """
    input: numpy file, index list, COSMIC file
    output: sigma file
    """
    cosmic_df = pd.read_csv(cosmicf, sep='\t')
    cosmic_head = list(cosmic_df)[1:]
    new_name=[]
    for item in cosmic_head:
        new_name.append(item[2]+item[4]+item[0]+item[-1])
    # convert to lower letter
    new_name = [letter.lower() for letter in new_name]
    # add tumor column
    mut = np.load(npyf, allow_pickle=True)
    sigma_df = pd.DataFrame(mut, columns=new_name)
    sigma_df['tumor'] = index_ls
    sigma_df.to_csv(sigmaf, sep=',', index=None)

def npy2sigma(npyf, idf, cosmicf,sigmaf):
    """
    input: numpy file, sample id file, COSMIC file
    output: sigma file
    """
    cosmic_df = pd.read_csv(cosmicf, sep='\t')
    cosmic_head = list(cosmic_df)[1:]
    new_name=[]
    for item in cosmic_head:
        new_name.append(item[2]+item[4]+item[0]+item[-1])
    # convert to lower letter
    new_name = [letter.lower() for letter in new_name]
    # add tumor column
    mut = np.load(npyf, allow_pickle=True)
    sigma_df = pd.DataFrame(mut, columns=new_name)
    id_df = pd.read_csv(idf, sep='\n',header=None)
    id_ls = [item[0] for item in list(id_df.values)]
    sigma_df['tumor'] = id_ls
    sigma_df.to_csv(sigmaf, sep=',', index=None)

def npy2our(npyf, cosmicf,ourf):
    """
    input: numpy file, sample id file, COSMIC file
    output: our tsv file - signature file
    """
    cosmic_df = pd.read_csv(cosmicf, sep='\t')
    cosmic_head = list(cosmic_df)[1:]
    mut = np.load(npyf, allow_pickle=True)
    our_df = pd.DataFrame(mut, columns=cosmic_head)
    id_ls = list(range(np.shape(mut)[0]))
    our_df['ID'] = id_ls
    new_order = ['ID'] + cosmic_head
    our_df = our_df.reindex(columns=new_order)
    our_df.to_csv(ourf, sep='\t', index=None)


if __name__ == "__main__":
    msk_dir = "/Users/yuexichen/Desktop/lrgr_file/mskfiles/jan_downsize"
    cosmicf =join(msk_dir, "cosmic-signatures.tsv")
    """
    cancer ="ov"
    ds_list = [2,5,10]
    p_list = [1,2]
    for dl in ds_list:
        for p in p_list:
            npyf = join(msk_dir,"mix_downsize/models/OV-ds%d-part%d-data.npy"%(dl, p))
            sigmaf = join(msk_dir, "ov-downsize%s_sigma_sbs_part%d.tsv"%(dl,p))
            all_index = np.load(join(msk_dir,"mix_downsize/indices/OV-ds%d-indices.npy"%dl))
            if p == 1:
                index_ls = all_index[:len(all_index) // 2]
            elif p==2:
                index_ls = all_index[len(all_index) // 2:]
            
            npy2sigma_nume(npyf, index_ls, cosmicf,sigmaf)
    
    #setting = ["clustered-nmf", "mix", "nmf"]
    """
    cancer = ["ov", "brca"]
    ds_ratio = ['003','006','009','012','015']
    for cc in cancer:
        for ds in ds_ratio:
            npyf = join(msk_dir, "%s-downsize%s_counts.npy"%(cc, ds))
            idf = join(msk_dir, "%s-downsize%s_sample_id.csv"%(cc, ds))
            sigmaf = join(msk_dir, "sigma-%s-downsize%s.tsv"%(cc, ds))
            npy2sigma(npyf, idf, cosmicf,sigmaf) 

    
    