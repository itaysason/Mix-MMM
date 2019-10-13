import pandas as pd
import numpy as np
from os.path import join
import json
import random

def select_msk(gene_f, ref_f, msk_locf):
    """
    Get MSK genes and distributions on chromosome
    INPUT:
    gene_f: gene files
    ref_f: reference file
    OUTPUT:
    msk_locf: genes with their chromosome regions
    """
    with open(gene_f) as in_f:
        gene_all = in_f.readlines()
    gene_ls = gene_all[3].rstrip().split('\t')[1:]
    assert(len(gene_ls) == 410)
    gloc_df = pd.read_csv(ref_f, sep='\t')    
    msk_loc = gloc_df[gloc_df['HGNC symbol'].isin(gene_ls)]
    msk_loc.to_csv(msk_locf, sep='\t', index=None)

def merge_by_chrom(msk_locf, region_js):
    """
    Ignore gene names but merge regions by chromosomes,
    INPUT:
    msk_locf: genes with their chromosome regions
    OUTPUT:
    region_js: chromosome with regions
    """
    msk_df = pd.read_csv(msk_locf, sep='\t')
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
    
    with open(region_js, 'w') as in_f:
        json.dump(region_dict, in_f)

def slim_wgs(wgsf, slim_wgsf):
    """
    only select chromo number, start/end positions to decrease file size
    INPUT:
    wgsf: whole genome MAF file
    OUTPUT:
    slim_wgsf: selected columns file
    """
    all_df = pd.read_csv(wgsf, sep='\t')
    #for SBS, start and end are the same position
    slim_df = all_df[['Patient', 'Chromosome', 'Start Position']]
    slim_df.to_csv(slim_wgsf, index=None)

def trap_wgs(region_js, slim_wgsf, trap_mutf, other_mutf):
    """
    if any mutation falls into such trap regions, record their row number, otherwise pass
    INPUT:
    region_js: chromosome with regions
    slim_wgs: selected columns file
    OUTPUT:
    trap_mutf: index of mutations in MSK regions
    other_mutf: index of mutations outside of MSK regions
    """
    slim_wg_df = pd.read_csv(slim_wgsf)
    with open(region_js) as in_f:
        region_dict =json.load(in_f)
    #trap mutations are those fall into MSK regions while other mutations are outside of the region
    trap_mut = []
    other_mut = []
    posi_ls = list(slim_wg_df['Start Position'])
    for i in range(len(posi_ls)):
        if i % 50000 == 0:
            print("Now it's %0.3f percentage"%(100.0*i/len(posi_ls)))
        try:
            chrom_region = region_dict[slim_wg_df['Chromosome'][i]]
        except:
            #ignore other mutations on other chromsomes
            chrom_region = [[0,0]]
        #append only one
        trap_ind = False
        for cr_tuple in chrom_region:
            #print("This is tuple in chrom region", cr_tuple, "of chromosome %s"%slim_wg_df['Chromosome'][i])
            if (int(posi_ls[i]) > int(cr_tuple[0])) and (int(posi_ls[i]) < int(cr_tuple[1])):
                trap_ind = True
            else:
                pass
        if trap_ind == False:
            other_mut.append(i)
        else:
            trap_mut.append(i)
    #rm duplicates
    other_mut = list(set(other_mut))
    with open(trap_mutf,'w') as out_f1:
        for tm in trap_mut:
            out_f1.write(str(tm) + '\n')
    with open(other_mutf,'w') as out_f2:
        for om in other_mut:
            out_f2.write(str(om) + '\n')     

def downsize(trap_mutf, keep_mutf, denominator, seed=1234):
    """
    a random generator selects mutations 1 out of 1000? Make sure the magnitude is comparable to MSK panels.
    INPUT: 
    trap_mutf: trapped mutations
    OUTPUT:
    keep_mutf: kept mutations
    """
    with open(trap_mutf) as in_f:
        #read as a list
        trap_ls = [line.rstrip('\n') for line in in_f.readlines()]
    print("We have %d trapped mutations"%len(trap_ls))
    random.seed(seed)
    keep_ls = []
    for i in range(len(trap_ls)):
       now = random.uniform(0,1) 
       if now < 1.0/denominator:
           keep_ls.append(trap_ls[i])
    print("We select %d mutations"%len(keep_ls))
    with open(keep_mutf,'w') as out_f:
        for kl in keep_ls:
            out_f.write(str(kl) + '\n')

def get_simulated_maf(wgsf, new_wgsf, keep_mutf):
    """
    extract such rows selected by downsizing,
    INPUT:
    wgsf: WGS file
    keep_mutf: mutations selected
    other_mutf: other mutations 
    OUPUT:
    new_wgsf: new WGS file
    """
    wgs_df = pd.read_csv(wgsf, sep='\t')
    with open(keep_mutf) as in_f1:
        keep_mut = [line.rstrip('\n') for line in in_f1.readlines()]
    with open(other_mutf) as in_f2:
        other_mut = [line.rstrip('\n') for line in in_f2.readlines()]
    all_mut = keep_mut
    wgs_df = pd.read_csv(wgsf, sep='\t')
    #number of patients
    patients = len(set(wgs_df['Patient']))
    print("we have %d patients"%patients)
    sele_df = wgs_df.iloc[all_mut]
    sele_df.to_csv(new_wgsf, sep='\t', index=None)

if __name__ == "__main__":
    #file names
    wgs_dir = "/Users/yuexichen/Downloads/lrgr_file/big_repo/mutation-signatures-data/mutations/ICGC/processed/standard"
    wgsf = join(wgs_dir, "standard.ICGC-BRCA-EU_BRCA_22.WGS.tsv")
    msk_dir =  "/Users/yuexichen/Downloads/lrgr_file/mskfiles/"
    gene_f = join(msk_dir, "data_gene_panel_impact410.txt") 
    ref_f = gloc_f = join(msk_dir, "mart_export.txt")
    msk_locf = join(msk_dir, 'msk_gene_loc.txt')
    slim_wgsf = join(msk_dir, 'slim_wgs.csv')
    region_js = join(msk_dir, 'region_dict.json')
    trap_wgsf = join(msk_dir, 'trap_wgs.csv')
    trap_mutf = join(msk_dir, 'trap_mut.csv')
    other_mutf = join(msk_dir, 'other_mut.csv')
    keep_mutf = join(msk_dir, 'keep_mut.csv')
    denominator = 10
    new_wgsf = join(msk_dir,'new_wgs_downsize%d.csv'%denominator)
    #Functions
    #select_msk(gene_f, ref_f, msk_locf)
    #merge_by_chrom(msk_locf, ref_f)
    #slim_wgs(wgsf, slim_wgsf)
    #trap_wgs(region_js, slim_wgsf, trap_mutf, other_mutf)
    #downsize(trap_mutf, keep_mutf, denominator, seed=1234)
    #get_simulated_maf(wgsf, new_wgsf, keep_mutf)


