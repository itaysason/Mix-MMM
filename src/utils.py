import json
import numpy as np
from src.models.Mix import Mix
import pandas as pd
import os


def save_json(file_name, dict_to_save):
    with open(file_name + '.json', 'w') as fp:
        json.dump(dict_to_save, fp)


def load_json(file_name):
    return json.load(open(file_name))


def category_smooth(ori_data, sv=1e-3):
    """
    check whether a category consists of all zeros, and add a small constant to smooth
    """
    # get column sums
    sum_ori_column = ori_data.sum(axis=0)
    zero_loc = np.where(sum_ori_column == 0)[0]
    if zero_loc.size == 0:
        print("No smooth needed")
    else:
        print("The following columns need smooth", zero_loc)
        ori_data = ori_data.astype(float)
        for zero_col in zero_loc:
            ori_data[:, zero_col] += sv
    return ori_data


def get_oncocode(threshold):
    """
    return all msk datasets names which have a count larger than the threshold
    """
    all_df = pd.read_csv("data/processed/oncotype_counts.txt", sep='\t')
    all_df['Counts'] = all_df['Counts'].astype(int)
    oncocode = all_df[all_df['Counts'] > threshold]['Oncotree']
    return oncocode


def get_active_sig(oncocode_ls):
    """
    input: a list of oncocode
    output: the index of mutation signatures involved with those oncocode based on Fig 3 of Alexandrov et al.
    """
    signature_dict = {'LUAD': [1, 2, 4, 5],
                      # IDC is a branch of BRCA
                      'IDC': [1, 2, 3, 8, 13],
                      # Colorectal
                      'COAD': [1, 6, 10],
                      # prostate
                      'PRAD': [1, 6],
                      # Pancreatic
                      'PAAD': [1, 2, 3, 6],
                      # Bladder
                      'BLCA': [1, 2, 5, 10, 13],
                      # Glioblastoma
                      'GBM': [1, 11],
                      # renal clear cell carcinoma (CCRCC) is a branch of kidney cancers
                      'CCRCC': [1, 6],
                      # SKCM is a branch of melanoma
                      'SKCM': [1, 7, 11],
                      # ILC is a branch of BRCA
                      'ILC': [1, 2, 3, 8, 13],
                      # Lung squamous
                      'LUSC': [2, 4, 5],
                      # Stomach
                      'STAD': [1, 2, 15, 17, 20, 21],
                      # READ is most similar to COAD
                      'READ': [1, 6, 10],
                      # unknow primary
                      'CUP': [1, 2],
                      # Gastro is not in the figure, use the unknown primary
                      'GIST': [1, 2],
                      # HGSOC is a branch of ovarian cancer
                      'HGSOC': [1, 3],
                      # IHCH is a branch of liver cancer
                      'IHCH': [1, 4, 6, 12, 16, 17],
                      # ESCA is most similar to stomach
                      'ESCA': [1, 2, 15, 17, 20, 21]}
    active_signatures = []
    for ol in oncocode_ls:
        tmp = signature_dict[ol]
        active_signatures.extend(tmp)
        # remove duplicates
    active_signatures = list(set(active_signatures))
    return active_signatures


def get_data(dataset, threshold=100):
    """
    threshold: filter out oncology types with few samples, i.e. fewer than the threshould.
    """
    if dataset == 'ICGC-BRCA':
        data = np.load('data/BRCA_counts.npy')
        active_signatures = [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]
    elif dataset == 'MSK-ALL':
        oncocode = get_oncocode(threshold)
        tmp_data_ls = []
        loaded_onco = []
        for oc in oncocode:
            dat_f = "data/processed/%s_counts.npy" % oc
            if not os.path.isfile(dat_f):
                Warning('%s is not loaded: file does not exist!' % dat_f)
            else:
                tmp_data = np.array(np.load(dat_f, allow_pickle=True), dtype=np.float64)
                tmp_data_ls.append(tmp_data)
                loaded_onco.append(oc)
        data = np.vstack(tmp_data_ls)
        print("Successfully loaded %d samples from %d datasets: %s" % (
            np.shape(data)[0], len(loaded_onco), ', '.join(loaded_onco)))
        active_signatures = get_active_sig(loaded_onco)
        print("Here are activate signatures", active_signatures)
    elif dataset == 'clustered-MSK-ALL':
        oncocode = get_oncocode(threshold)
        tmp_data_ls = []
        loaded_onco = []
        for oc in oncocode:
            dat_f = "data/processed/%s_counts.npy" % oc
            if not os.path.isfile(dat_f):
                Warning('%s is not loaded: file does not exist!' % dat_f)
            else:
                tmp_data = np.array(np.load(dat_f, allow_pickle=True), dtype=np.float64)
                tmp_data_ls.append(tmp_data.sum(0, keepdims=True))
                loaded_onco.append(oc)
        data = np.vstack(tmp_data_ls)
        print("Successfully loaded %d samples from %d datasets: %s" % (
            np.shape(data)[0], len(loaded_onco), ', '.join(loaded_onco)))
        active_signatures = get_active_sig(loaded_onco)
        print("Here are activate signatures", active_signatures)
    elif dataset == 'MSK-filtered':
        min_num_mutations = 5
        min_num_samples = 50
        oncocode = get_oncocode(0)
        tmp_data_ls = []
        loaded_onco = []
        cancer_clusters = []
        for oc in oncocode:
            dat_f = "data/processed/%s_counts.npy" % oc
            if not os.path.isfile(dat_f):
                Warning('%s is not loaded: file does not exist!' % dat_f)
            else:
                tmp_data = np.array(np.load(dat_f, allow_pickle=True), dtype=np.float64)
                tmp_data = tmp_data[tmp_data.sum(1) >= min_num_mutations]
                if len(tmp_data) >= min_num_samples:
                    tmp_data_ls.append(tmp_data)
                    loaded_onco.append(oc)
                    cancer_clusters.extend([oc] * len(tmp_data))
        data = np.vstack(tmp_data_ls)
        print("Successfully loaded %d samples from %d datasets: %s" % (
            np.shape(data)[0], len(loaded_onco), ', '.join(loaded_onco)))
        active_signatures = get_active_sig(loaded_onco)
        print("Here are activate signatures", active_signatures)
    elif dataset == 'MSK-filtered2':
        min_num_mutations = 5
        min_num_samples = 50
        oncocode = get_oncocode(0)
        tmp_data_ls = []
        loaded_onco = []
        cancer_clusters = []
        for oc in oncocode:
            dat_f = "data/processed/%s_counts.npy" % oc
            if not os.path.isfile(dat_f):
                Warning('%s is not loaded: file does not exist!' % dat_f)
            else:
                tmp_data = np.array(np.load(dat_f, allow_pickle=True), dtype=np.float64)
                tmp_data = tmp_data[tmp_data.sum(1) >= min_num_mutations]
                if len(tmp_data) >= min_num_samples:
                    tmp_data_ls.append(tmp_data)
                    loaded_onco.append(oc)
                    cancer_clusters.extend([oc] * len(tmp_data))
        data = np.vstack(tmp_data_ls)
        print("Successfully loaded %d samples from %d datasets: %s" % (
            np.shape(data)[0], len(loaded_onco), ', '.join(loaded_onco)))
        active_signatures = get_active_sig(loaded_onco)
        print("Here are activate signatures", active_signatures)
    elif dataset == 'TCGA-OV':
        data = np.load('data/WXS-TCGA-OV/wxs-ov-all_counts.npy').astype('float')
        active_signatures = [1, 3, 5]
    elif dataset == 'ICGC-BRCA-ds100':
        data = np.load('data/WGS-NikZainal-BRCA/downsize_100_counts.npy').astype('float')
        active_signatures = [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]
    elif dataset == 'ICGC-BRCA-ds10':
        data = np.load('data/WGS-NikZainal-BRCA/downsize_10_counts.npy').astype('float')
        active_signatures = [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]
    elif dataset == 'new-BRCA':
        data = np.load('data/WGS-NikZainal-BRCA/new-wgs-brca-d500_counts.npy').astype('float')
        active_signatures = [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]
    elif dataset == 'TCGA-OV-msk-region':
        data = np.load('data/WXS-TCGA-OV/wxs-ov-msk-region_counts.npy').astype('float')
        active_signatures = [1, 3, 5]
    elif dataset == 'new-OV':
        data = np.load('data/WXS-TCGA-OV/new-wxs-ov-d10_counts.npy').astype('float')
        active_signatures = [1, 3, 5]
    else:
        raise ValueError('No such dataset {}'.format(dataset))

    active_signatures = np.array(active_signatures) - 1
    return data, active_signatures


def get_model(parameters):
    parameters = {key: np.array(val) for key, val in parameters.items()}
    num_clusters = len(parameters['w'])
    num_topics = len(parameters['e'])
    model = Mix(num_clusters, num_topics, parameters)
    return model


def get_cosmic_signatures(dir_path='data/signatures/COSMIC/cosmic-signatures.tsv'):
    return pd.read_csv(dir_path, sep='\t', index_col=0).values


if __name__ == "__main__":
    get_data('MSK-ALL')
