import pandas as pd
import numpy as np
from os.path import join
import json
import random
from collections import Counter

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

def rawmsk2dict(raw_mskloc, region_js):
    """
    input: raw msk gene location file
    output: a json file, keys are chromsomes, values are a list of start/end, the same format as "merge_by_chrom" function
    """
    excel_df = pd.read_excel(raw_mskloc, 'SupplTable_PanelDesign.txt', index_col=None, skiprows=3)
    all_chrom = list(set(excel_df['Chr']))
    region_dict = {}
    for chrom in all_chrom:
        now_df = excel_df[excel_df['Chr']==chrom]
        start_loc = pd.to_numeric(now_df["start"]).tolist()
        end_loc = pd.to_numeric(now_df["stop"]).tolist()
        loc_ls = list(zip(start_loc, end_loc))
        region_dict[chrom] = loc_ls

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

def trap_wgs(region_js, slim_wgsf, trap_mutf):
    """
    if any mutation falls into such trap regions, record their row number, otherwise pass
    INPUT:
    region_js: chromosome with regions
    slim_wgs: selected columns file
    OUTPUT:
    trap_mutf: index of mutations in MSK regions
    """
    slim_wg_df = pd.read_csv(slim_wgsf)
    with open(region_js) as in_f:
        region_dict =json.load(in_f)
    #trap mutations are those fall into MSK regions while other mutations are outside of the region
    trap_mut = []
    posi_ls = list(slim_wg_df['Start Position'])
    for i in range(len(posi_ls)):
        if i % 500000 == 0:
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
        if trap_ind == True:
            # keep the row index
            trap_mut.append(i)
            #if slim_wg_df['Patient'][i] in trap_mut_dict.keys():
            #    trap_mut_dict[slim_wg_df['Patient'][i]].append(i)
            #else:
            #    trap_mut_dict[slim_wg_df['Patient'][i]] = [i]
    with open(trap_mutf,'w') as out_f:
        for tm in trap_mut:
            out_f.write(str(tm) + '\n')    

def get_trap_maf(wgsf, new_wgsf, trap_mutf):
    """
    extract such rows selected by downsizing,
    INPUT:
    wgsf: WGS file
    trap_mutf: mutations selected 
    OUPUT:
    new_wgsf: new WGS file
    """
    wgs_df = pd.read_csv(wgsf, sep='\t')
    with open(trap_mutf) as in_f1:
        trap_mut = [line.rstrip('\n') for line in in_f1.readlines()]
    
    all_mut = [int(items) for items in trap_mut]
    wgs_df = pd.read_csv(wgsf, sep='\t')
    #number of patients
    patients_num = len(set(wgs_df['Patient']))
    print("We have %d patients"%patients_num)
    sele_df = wgs_df.iloc[all_mut]
    print("We have %0.2f mutations per patients"%(len(sele_df.index)/patients_num))
    sele_df.to_csv(new_wgsf, sep='\t', index=None)

def downsize(full_cnf, ds_cnf, id_f, ds_mean, seed=1234):
    """
    Sample from a Poisson distribution with lambda = downsample_mean; 
    for each patient, sample a k, if the number of mutations in that patients is smaller than k,
    then keep all mutations in that patients.  
    INPUT: 
    full_cnf: full 96-mutation count file
    OUTPUT:
    ds_cnf: downsampled 96-mutation numpy file
    id_f: associated sample id
    """
    #random.seed(seed)
    np.random.seed(seed)
    mut_pd = pd.read_csv(full_cnf, sep='\t')
    mut_val = mut_pd.values[:,1:]
    patient_num = np.shape(mut_val)[0]
    ds_num = np.random.poisson(ds_mean, patient_num)
    # extend count to num list
    ds_mut_all = []
    for i in range(patient_num):
        tmp_full = []
        tmp_ds = []
        for j in range(96):
            # e.g. flatten count file
            tmp_full.extend([j] * int(mut_val[i][j]))
        if (len(tmp_full) < ds_num[i]):
            tmp_ds = tmp_full
        # we want to keep all patients
        elif (ds_num[i]==0):
            tmp_ds = random.sample(tmp_full, 1)
        else:  
            tmp_ds = random.sample(tmp_full, ds_num[i])
        # e.g. [[1,1,1],[2,3,3]] to [{1:3},[{2:1},{3:2}]]
        dict_tmpds = dict(Counter(tmp_ds))
        ds_mut_all.append(dict_tmpds)
    ds_npy = np.zeros([patient_num, 96])
    for i in range(patient_num):
        dma = ds_mut_all[i]
        now_keys = list(dma.keys())
        for nk in now_keys:
            ds_npy[i][nk] = dma[nk]
    
    # output
    np.save(ds_cnf, ds_npy)
    mut_name = list(mut_pd.values[:,0])
    with open(id_f,'w') as nf:
        for mn in mut_name:
            nf.write(mn + '\n')

    # summary statistics
    print("Before sampling, we have %d mutations, on average we have %0.2f mutations per patient"%(np.sum(mut_val), np.sum(mut_val)/patient_num))
    print("After sampling, we have %d mutations, on average we have %0.2f mutations per patient"%(np.sum(ds_npy), np.sum(ds_npy)/patient_num))
