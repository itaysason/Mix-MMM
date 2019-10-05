import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})
rcParams.update({'font.size': 12})
import argparse
import os
import numpy as np
import pandas as pd
from os.path import join


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-ond','--oncotree_dir', default='data/processed')
    parser.add_argument('-ot','--oncotree_code', type=list, help='a list of oncotree code',default=['PRAD', 'PAAD','BLCA'])
    parser.add_argument('-otd','--out_dir',default='figures/')
    args = parser.parse_args()
    if os.path.isdir(args.out_dir):
        pass
    else:
        os.makedirs(args.out_dir)
    onco_df = pd.DataFrame(columns=['oncotree code','Number of SBS'])
    mut_ls = []
    onco_ls = []
    for code in args.oncotree_code:
        code_f = join(args.oncotree_dir, '%s_counts.npy'%code)
        if not os.path.isfile(code_f):
            raise FileNotFoundError("File % not found"%code_f)
        else:
            mut_np = np.load(code_f, allow_pickle=True)
            mut_sum = mut_np.sum(axis=1)
            mut_ls.extend(mut_sum)
            onco_ls.extend([code]*len(mut_sum))
    onco_df['Number of SBS'] = mut_ls
    onco_df['Oncotree code'] = onco_ls
    ax = sns.boxplot(x="Oncotree code", y="Number of SBS", data=onco_df, showfliers = False)
    ax.set_yscale('log')
    ax = sns.swarmplot(x="Oncotree code", y="Number of SBS", data=onco_df, color=".25")
    plt.savefig(join(args.out_dir, 'boxplot-%s.pdf'%('-'.join(args.oncotree_code))))
    plt.show()
