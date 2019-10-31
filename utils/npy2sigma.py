import pandas as pd
import numpy as np
from os.path import join

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



if __name__ == "__main__":
    msk_dir = "/Users/yuexichen/Downloads/lrgr_file/mskfiles"
    cancer ="brca"
    npyf= join(msk_dir,"%s-all_counts.npy"%cancer)
    idf = join(msk_dir, "%s-all_sample_id.txt"%cancer)
    cosmicf =join(msk_dir, "cosmic-signatures.tsv")
    sigmaf = join(msk_dir, "%s-all_sigma_sbs.tsv"%cancer)
    npy2sigma(npyf, idf, cosmicf, sigmaf)
