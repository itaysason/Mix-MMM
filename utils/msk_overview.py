import pandas as pd
import numpy as np
import argparse
import os
from os.path import join
from collections import Counter, OrderedDict
import json

parser = argparse.ArgumentParser()
parser.add_argument('-rf','--raw_f', help='the clinical data file',default='data/raw/data_clinical_sample.txt')
parser.add_argument('-of','--out_f', help='output file', default='data/processed/oncotype_counts.txt')
args = parser.parse_args()

clinic_df = pd.read_csv(args.raw_f, sep='\t')
#ignore headers
onco_ls = list(clinic_df['Oncotree Code'])
#return the occurence of each onco type and the number of such occurences, save the result to a file
onco_dict = Counter(onco_ls)
onco_ordered = OrderedDict(sorted(onco_dict.items(), key=lambda t: t[1], reverse=True))
with open(args.out_f,'w') as f:
    f.write( "{}\t{}\n".format('Oncotree','Counts'))
    for k,v in onco_ordered.items():
        f.write( "{}\t{}\n".format(k,v))


