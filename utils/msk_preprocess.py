import numpy as np
import pandas as pd
import argparse
import os
from os.path import join

parser = argparse.ArgumentParser()
parser.add_argument('-rd','--raw_dir',help='tsv file folders', default='data/raw/')
parser.add_argument('-ot','--onco_tree', help='onco tree code',type=str, default='LUAD')
parser.add_argument('-od','--out_dir', help='processed numpy data dir', default='data/processed')
args = parser.parse_args()

"""
#input: SBS 96 tsv files with mutation categories as header and user ids as row names. 
#output: 1. mutation profiles in numpy formats, 
#        2. user id in txt format.
"""

raw_f = join(args.raw_dir, 'counts.cBioPortal-msk-impact-2017_%s_6800765.TARGETED.SBS-96.tsv'%args.onco_tree)
mut_pd = pd.read_csv(raw_f, sep='\t')
mut_val = mut_pd.values[:,1:]
print("Shape", np.shape(mut_val))
if os.path.isdir(args.out_dir):
    pass
else:
    os.makedirs(args.out_dir)
out_f = join(args.out_dir, '%s_counts'%args.onco_tree)
np.save(out_f,mut_val)
name_f = join(args.out_dir, '%s_sample_id.txt'%args.onco_tree)
mut_name = list(mut_pd.values[:,0])
with open(name_f,'w') as nf:
    for mn in mut_name:
        nf.write(mn + '\n')






