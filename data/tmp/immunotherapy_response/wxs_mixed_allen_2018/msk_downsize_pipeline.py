import pandas as pd
import numpy as np
from os.path import join
import json
import random
from collections import Counter
import explosig_data as ed

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

def merge_by_chrom(msk_locf, region_js, col1,col2,col3):
    """
    Ignore gene names but merge regions by chromosomes,
    INPUT:
    msk_locf: genes with their chromosome regions
    OUTPUT:
    region_js: chromosome with regions
    PARAMS:
    column names, col1, 2, 3
    """
    msk_df = pd.read_csv(msk_locf, sep='\t')
    #print(msk_df)
    chrom_name = list(msk_df[col1])
    start_bp = list(msk_df[col2])
    end_bp = list(msk_df[col3])
    region_dict = dict()
    for i in range(len(chrom_name)):
        if chrom_name[i].isdigit() or chrom_name[i] == 'X':
            if chrom_name[i] in region_dict.keys():
                region_dict[chrom_name[i]].append((start_bp[i], end_bp[i])) 
            else:
                region_dict[chrom_name[i]] = [(start_bp[i], end_bp[i])]
        else:
            pass
    #print(region_dict)
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

def standard2count(maf, count_f):
    #if dat_type == "icgc":
    #    data_container = ed.standardize_ICGC_ssm_file(maf)
    #elif dat_type == 'tcga':
    #    data_container = ed.standardize_TCGA_maf_file(maf)
    #data_container.extend_df().to_counts_df('SBS_96', ed.categories.SBS_96_category_list())
    #counts_df = data_container.counts_dfs['SBS_96']
    #counts_df.to_csv(count_f, sep='\t', index=None)
    ssm_df = pd.read_csv(maf, sep='\t')
    data_container = ed.SimpleSomaticMutationContainer(ssm_df)
    data_container.extend_df().to_counts_df('SBS_96', ed.categories.SBS_96_category_list())
    counts_df = data_container.counts_dfs['SBS_96']
    counts_df.to_csv(count_f, sep='\t')

def extra_sampling(full_cnf, filter_cnf, ex_cnf, extra, seed):
    """
    Sample with the EXTRA times of mutations as before for each patient  
    INPUT: 
        full_cnf: full 96 count file
        filter_cnf: filtered 96 count file
        ex_cnf: output 96 count file
        extra: a float
        seed: random seed
    OUTPUT:
        ex_cnf: extra 96-mutation numpy file from original full file
    """
    random.seed(seed)
    np.random.seed(seed)
    mut_pd = pd.read_csv(full_cnf, sep='\t')
    full_id = list(mut_pd.values[:,1])
    mut_val = mut_pd.values[:,1:].astype(int)
    #oriignal patient number
    patient_num = np.shape(mut_val)[0]

    filter_pd = pd.read_csv(filter_cnf, sep='\t')
    # the first column
    filter_id = list(filter_pd.values[:,0])
    #print(filter_id)
    filter_val = filter_pd.values[:,1:].astype(int)
    filter_num_ls = np.sum(filter_val,axis=1).tolist()
    # some samples are filtered out
    ds_num_dict = {}
    for m in range(len(filter_id)):
        ds_num_dict[filter_id[m]] = int(filter_num_ls[m]*extra)
    
    # extend count to num list
    ds_mut_all = []
    for i in range(len(filter_id)):
        tmp_full = []
        tmp_ds = []
        extra_cnt = ds_num_dict[filter_id[i]] 
        #print(extra_cnt)
        for j in range(96):
            # e.g. flatten count file for sampling
            tmp_full.extend([j] * int(mut_val[i][j]))
        if (len(tmp_full) < extra_cnt):
            tmp_ds = tmp_full
        else:  
            tmp_ds = random.sample(tmp_full, extra_cnt)
        # e.g. [[1,1,1],[2,3,3]] to [{1:3},[{2:1},{3:2}]]
        dict_tmpds = dict(Counter(tmp_ds))
        ds_mut_all.append(dict_tmpds)
    
    ds_npy = np.zeros([len(filter_id), 96])
    for i in range(len(filter_id)):
        dma = ds_mut_all[i]
        now_keys = list(dma.keys())
        for nk in now_keys:
            ds_npy[i][nk] = dma[nk]
    
    mut_name = list(filter_pd.values[:,0])
    # save to tsv
    new_mut_pd = pd.DataFrame(data=ds_npy, columns=list(filter_pd)[1:],index=mut_name)
    new_mut_pd.to_csv(ex_cnf, sep='\t')

    # summary statistics
    print("Before sampling, we have %d patients, %d mutations, on average we have %0.2f mutations per patient"%(patient_num, np.sum(mut_val), np.sum(mut_val)/patient_num))
    print("After sampling extra, we have %d patients, %d mutations, on average we have %0.2f mutations per patient"%(len(filter_id), np.sum(ds_npy), np.sum(ds_npy)/patient_num))


def combine_f(extra_cnf, filter_cnf, combine_cnf):
    """
    just combine two dataframe together to combine_cnf
    """
    extra_df = pd.read_csv(extra_cnf, sep='\t')
    extra_cn = extra_df.values[:,1:]
    filter_cn = pd.read_csv(filter_cnf, sep='\t').values[:,1:]
    combine_cn = extra_cn + filter_cn
    id_index = pd.read_csv(extra_cnf, sep='\t').values[:,0]
    combine_df = pd.DataFrame(data=combine_cn, columns=list(extra_df)[1:],index=id_index)
    combine_df.to_csv(combine_cnf, sep='\t')

def full_sampling(combine_cnf, ds_npyf, id_f, ds_tsv, ds_mean, threshold, seed):
    """
    Sample from a Poisson distribution with lambda = downsample_mean; 
    for each patient, sample a k, if the number of mutations in that patients is smaller than k,
    then keep all mutations in that patients.  
    INPUT: 
        combine_cnf: full 96-mutation count file
    OUTPUT:
        ds_npyf: downsampled 96-mutation numpy
        id_f: associated sample id
        ds_tsv: downsampled 96-mut tsv
    PARAM:
        ds_mean: lambda for poisson
        threshold: ignore samples with fewer mutations thanthe threshold
        seed: random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    mut_pd = pd.read_csv(combine_cnf, sep='\t')
    all_sum =  mut_pd.values[:,1:].sum(axis=1).tolist()
    #filter by threshold

    mut_pd.loc[:,'Total'] = mut_pd.sum(axis=1)
    print("Before sampling, we have %d patient, %0.2f mutations per patient, max: %d, min: %d"%(len(mut_pd.index),sum(all_sum)/len(all_sum), max(all_sum), min(all_sum)))

    mut_pd = mut_pd[mut_pd['Total']>threshold]
    mut_pd = mut_pd.drop(columns=['Total'])

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


    # output for mix
    np.save(ds_npyf, ds_npy)
    mut_name = list(mut_pd.values[:,0])
    with open(id_f,'w') as nf:
        for mn in mut_name:
            nf.write(mn + '\n')
    #print(len(list(mut_pd)))
    #print(np.shape(ds_npy))
    # save to tsv
    new_mut_pd = pd.DataFrame(data=ds_npy, columns=list(mut_pd)[1:],index=mut_name)
    new_mut_pd.to_csv(ds_tsv, sep='\t')

    # summary statistics
    all_sum =  new_mut_pd.values[:,1:].sum(axis=1).tolist()
    print("After sampling, we have %d patient, %0.2f mutations per patient, max: %d, min: %d"%(len(mut_pd.index),sum(all_sum)/len(all_sum), max(all_sum), min(all_sum)))


