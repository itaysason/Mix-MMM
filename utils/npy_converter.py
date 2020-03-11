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

def npy2our(npyf, cosmicf, ourf):
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

def many_npy2our(onco_list, data_dir, cosmicf, ourf):
    """
    input: 
        onco_list: code of cancer types 
        data_dir: a directory of numpy files, sample id file
        cosmicf: the standard file (the header could be reused)
        ourf: concatenated tsv file
    """
    data_ls = []
    id_ls = []
    for ol in onco_list:
        dat_f = join(data_dir, ol + "_counts.npy")
        tmp_data = np.array(np.load(dat_f, allow_pickle=True), dtype=np.float64)
        data_ls.append(tmp_data)
        
        tmp_idf = join(data_dir, ol + "_sample_id.txt")
        with open(tmp_idf) as in_f:
            tmp_id = in_f.read().splitlines()
        id_ls.extend(tmp_id)
    
    all_data = np.vstack(data_ls)
    cosmic_df = pd.read_csv(cosmicf, sep='\t')
    cosmic_head = list(cosmic_df)[1:]

    our_df = pd.DataFrame(all_data, columns=cosmic_head)
    our_df['ID'] = id_ls
    new_order = ['ID'] + cosmic_head
    our_df = our_df.reindex(columns=new_order)
    our_df.to_csv(ourf, sep='\t', index=None)

if __name__ == "__main__":
    msk_dir = "/Users/yuexichen/Desktop/lrgr_file/mskfiles/jan_downsize/sparse"
    cosmicf =join(msk_dir, "cosmic-signatures.tsv")
    #data_dir = "/Users/yuexichen/Desktop/LRGR/Repository/Mix-MMM/data/processed/msk_impact"
    data_dir = '/Users/yuexichen/Desktop/LRGR/Repository/Mix-MMM/data/panel_downsize'
    ourf = "/Users/yuexichen/Desktop/LRGR/Repository/Mix-MMM/data/processed/msk_impact/all_msk.tsv"
    onco_list = ["LUAD","IDC","COAD","PRAD","PAAD","BLCA","GBM","CCRCC","SKCM",
    "ILC","LUSC","STAD","READ","CUP","GIST","HGSOC","IHCH", "ESCA"]
    #many_npy2our(onco_list, data_dir, cosmicf, ourf)
    pref=['BRCA-panel-full-part','OV-panel-full-part']
    part=['1','2']
    for pr in pref:
        for pt in part:
            npyf= join(data_dir, pr + pt + '_counts.npy')
            idf= join(data_dir, pr + pt + '_sample_id.csv')
            sigmaf= join(msk_dir, 'sigma-'+ pr + pt + '.tsv')
            npy2sigma(npyf, idf, cosmicf,sigmaf)


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
    """
    cancer = ["ov", "brca"]
    ds_ratio = ['003','006','009','012','015']
    #ds_ratio = ['018','021','024','027']
    for cc in cancer:
        for ds in ds_ratio:
            npyf = join(msk_dir, "%s-downsize%s_counts.npy"%(cc, ds))
            idf = join(msk_dir, "%s-downsize%s_sample_id.csv"%(cc, ds))
            sigmaf = join(msk_dir, "sigma-%s-downsize%s.tsv"%(cc, ds))
            npy2sigma(npyf, idf, cosmicf,sigmaf) 
    """


    
    