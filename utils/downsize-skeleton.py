import pandas as pd
import numpy as np
from os.path import join
import json

def select_msk(raw_dir):
    """
    Get MSK genes and distributions on chromosome
    """
    gene_ls_f = join(raw_dir, "data_gene_panel_impact410.txt") 
    with open(gene_ls_f) as in_f:
        gene_all = in_f.readlines()
    gene_ls = gene_all[3].rstrip().split('\t')[1:]
    assert(len(gene_ls) == 410)
    gloc_f = join(raw_dir, "mart.txt")
    gloc_df = pd.read_csv(gloc_f, sep='\t')    
    msk_loc = gloc_df[gloc_df['HGNC symbol'].isin(gene_ls)]
    #print(msk_loc)
    out_f = join(raw_dir, "msk_gene_loc.txt")
    msk_loc.to_csv(out_f, sep='\t', index=None)

def merge_by_chrom(msk_loc, raw_dir):
    """
    Ignore gene names but merge regions by chromosomes,
    return trap regions per chromosome.
    """
    msk_df = pd.read_csv(join(raw_dir, msk_loc), sep='\t')
    chrom_name = list(msk_df['Chromosome/scaffold name'])
    start_bp = list(msk_df['Gene start (bp)'])
    end_bp = list(msk_df['Gene end (bp)'])
    region_dict = dict()
    for i in range(len(chrom_name)):
        if chrom_name[i].isdigit() or chrom_name[i] == 'X':
            if chrom_name[i] in region_dict.keys():
                region_dict[chrom_name[i]].append((start_bp[i], end_bp[i])) 
            else:
                region_dict[chrom_name[i]] = [(start_bp[i], end_bp[i])]
        else:
            pass
    
    with open(join(raw_dir, 'region_dict.json'), 'w') as in_f:
        json.dump(region_dict, in_f)

    return region_dict
    

def slim_wgs(raw_wgs):
    """
    only select chromo number, start/end positions to decrease file size
    """


def trap_wgs():
    """
    if any mutation falls into such trap regions, record their row number, otherwise pass
    """

def downsize(denominator=1000):
    """
    a random generator selects mutations 1 out of 1000? Make sure the magnitude is comparable to MSK panels.
    return: a list of indexes
    """

def get_simulated_maf(raw_wgs):
    """
    only extract such rows selected by downsizing
    """

if __name__ == "__main__":
    raw_dir =  "/Users/yuexichen/Downloads/lrgr_file/mskfiles/"
    msk_loc = "msk_gene_loc.txt"
    merge_by_chrom(msk_loc, raw_dir)
    



