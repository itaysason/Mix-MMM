import json
import numpy as np
from src.models.Mix import Mix
import pandas as pd
import os
from src.constants import ROOT_DIR


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
        print("No smoothing needed")
    else:
        print("The following columns need smoothing", zero_loc)
        ori_data = ori_data.astype(float)
        for zero_col in zero_loc:
            ori_data[:, zero_col] += sv
    return ori_data


def get_oncocode(threshold):
    """
    return all msk datasets names which have a count larger than the threshold
    """
    all_df = pd.read_csv(os.path.join(ROOT_DIR, "data/MSK-processed/oncotype_counts.txt"), sep='\t')
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
        data = np.load(os.path.join(ROOT_DIR, 'data/ICGC-BRCA/BRCA_counts.npy'))
        active_signatures = [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]
    elif dataset == 'TCGA-OV':
        data = np.load(os.path.join(ROOT_DIR, 'data/TCGA-OV/OV_counts.npy')).astype('float')
        active_signatures = [1, 3, 5]
    elif dataset == 'MSK-ALL':
        oncocode = get_oncocode(threshold)
        tmp_data_ls = []
        loaded_onco = []
        onco_num_mutations = []
        for oc in oncocode:
            dat_f = os.path.join(ROOT_DIR, "data/MSK-processed/%s_counts.npy") % oc
            if not os.path.isfile(dat_f):
                Warning('%s is not loaded: file does not exist!' % dat_f)
            else:
                tmp_data = np.array(np.load(dat_f, allow_pickle=True), dtype=np.float64)
                tmp_data_ls.append(tmp_data)
                loaded_onco.append(oc)
                onco_num_mutations.append(tmp_data.sum())
        data = np.vstack(tmp_data_ls)
        print("Successfully loaded %d samples from %d datasets: %s" % (np.shape(data)[0], len(loaded_onco), ', '.join(loaded_onco)))
        active_signatures = get_active_sig(loaded_onco)
        print("Here are activate signatures", active_signatures)
    elif dataset == 'clustered-MSK-ALL':
        oncocode = get_oncocode(threshold)
        tmp_data_ls = []
        loaded_onco = []
        for oc in oncocode:
            dat_f = os.path.join(ROOT_DIR, "data/MSK-processed/%s_counts.npy") % oc
            if not os.path.isfile(dat_f):
                Warning('%s is not loaded: file does not exist!' % dat_f)
            else:
                tmp_data = np.array(np.load(dat_f, allow_pickle=True), dtype=np.float64)
                tmp_data_ls.append(tmp_data.sum(0, keepdims=True))
                loaded_onco.append(oc)
        data = np.vstack(tmp_data_ls)
        active_signatures = get_active_sig(loaded_onco)
    elif 'BRCA' in dataset and 'ds' in dataset and 'part' in dataset:
        ds_size = dataset.split('-')[1][2:]
        data = np.load(os.path.join(ROOT_DIR, 'data/ICGC-BRCA/downsampled/brca-downsize{}_counts.npy'.format(ds_size.zfill(3))))
        active_signatures = [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]
    elif 'OV' in dataset and 'ds' in dataset and 'part' in dataset:
        ds_size = dataset.split('-')[1][2:]
        data = np.load(os.path.join(ROOT_DIR, 'data/TCGA-OV/downsampled/ov-downsize{}_counts.npy'.format(ds_size.zfill(3))))
        active_signatures = [1, 3, 5]
    elif 'BRCA' in dataset and 'panel' in dataset:
        filtered_df = pd.read_csv(os.path.join(ROOT_DIR, "data/ICGC-BRCA/panel-BRCA-WGS_counts.tsv"), sep='\t')
        if 'full' in dataset:
            all_df = pd.read_csv(os.path.join(ROOT_DIR, "data/ICGC-BRCA/counts.ICGC-BRCA-EU_BRCA_22.WGS.SBS-96.tsv"), sep='\t')
            filtered_samples = np.array(filtered_df)[:, 0].astype('str')
            all_samples = np.array(all_df)[:, 0].astype('str')
            indices = [np.where(s == all_samples)[0][0] for s in filtered_samples]
            data = np.array(all_df)[indices, 1:].astype('float')
        elif 'hrd' in dataset:
            all_df = pd.read_csv(os.path.join(ROOT_DIR, "data/ICGC-BRCA/icgc_brca_hrd.tsv"), sep='\t')
            filtered_samples = np.array(filtered_df)[:, 0].astype('str')
            all_samples = np.array(all_df)[:, 1].astype('str')
            indices = [np.where(s == all_samples)[0][0] for s in filtered_samples]
            data = np.array(all_df)[indices, 2:].astype('float')
        else:
            data = np.array(filtered_df)[:, 1:].astype('float')
        active_signatures = [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]
    elif 'OV' in dataset and 'panel' in dataset:
        filtered_df = pd.read_csv(os.path.join(ROOT_DIR, "data/TCGA-OV/panel-OV-WXS_counts.tsv"), sep='\t')
        if 'full' in dataset:
            all_df = pd.read_csv(os.path.join(ROOT_DIR, "data/TCGA-OV/counts.TCGA-OV_OV_mc3.v0.2.8.WXS.SBS-96.tsv"), sep='\t')
            filtered_samples = np.array(filtered_df)[:, 0].astype('str')
            all_samples = np.array(all_df)[:, 0].astype('str')
            indices = [np.where(s == all_samples)[0][0] for s in filtered_samples]
            data = np.array(all_df)[indices, 1:].astype('float')
        else:
            data = np.array(filtered_df)[:, 1:].astype('float')
        active_signatures = [1, 3, 5]
    elif 'nature2019' in dataset:
        if 'full' in dataset:
            path = os.path.join(ROOT_DIR, 'data/nature2019/counts.staaf2019_BRCA_WGS.tsv')
        elif 'panel' in dataset:
            path = os.path.join(ROOT_DIR, 'data/nature2019/filter-staaf2019_BRCA-WGS_counts.tsv')
        elif 'labels' in dataset:
            path = os.path.join(ROOT_DIR, 'data/nature2019/labels.tsv')
        elif 'sigma-exp' in dataset:
            path = os.path.join(ROOT_DIR, 'data/nature2019/SigMA_output.tsv')
        else:
            raise ValueError('No such dataset {}'.format(dataset))
        df = pd.read_csv(path, sep='\t')
        df_samples = np.array(df)[:, 0].astype('str')
        samples = np.load(os.path.join(ROOT_DIR, 'data/nature2019/nature2019_samples.npy'))
        indices = [np.where(s == df_samples)[0][0] for s in samples]
        data = np.array(df)[indices, 1:].astype('float')
        active_signatures = [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]
    elif 'msk2018-LUAD' == dataset:
        metadata_path = os.path.join(ROOT_DIR, 'data/immunotherapy_response/panel_nsclc_pd1_msk_2018/labels_nsclc_pd1_2018.tsv')
        df = pd.read_csv(metadata_path, sep='\t')
        df = df[df['oncotree_code'] == 'LUAD']
        df = df[df['treatment_type'] == 'Monotherapy']
        df = df[df['durable_clinical_benefit'] != 'NE']
        samples = df['patient_id'].values
        counts_path = os.path.join(ROOT_DIR, 'data/immunotherapy_response/panel_nsclc_pd1_msk_2018/count96_nsclc_pd1_msk_2018.tsv')
        df = pd.read_csv(counts_path, sep='\t')
        count_samples = df['Unnamed: 0'].values
        new_samples = []
        indices = []
        for s in samples:
            for i, r in enumerate(count_samples):
                if s in r:
                    indices.append(i)
                    new_samples.append(s)
                    continue
        data = np.array(df)[indices, 1:].astype('float')
        active_signatures = [1, 2, 4, 5, 6, 13, 17]
    elif 'msk2018-LUAD-labels' == dataset:
        metadata_path = os.path.join(ROOT_DIR, 'data/immunotherapy_response/panel_nsclc_pd1_msk_2018/labels_nsclc_pd1_2018.tsv')
        df = pd.read_csv(metadata_path, sep='\t')
        df = df[df['oncotree_code'] == 'LUAD']
        df = df[df['treatment_type'] == 'Monotherapy']
        df = df[df['durable_clinical_benefit'] != 'NE']
        samples = df['patient_id'].values
        counts_path = os.path.join(ROOT_DIR, 'data/immunotherapy_response/panel_nsclc_pd1_msk_2018/count96_nsclc_pd1_msk_2018.tsv')
        df = pd.read_csv(counts_path, sep='\t')
        count_samples = df['Unnamed: 0'].values
        new_samples = []
        indices = []
        for s in samples:
            for i, r in enumerate(count_samples):
                if s in r:
                    indices.append(i)
                    new_samples.append(s)
                    continue
        metadata_path = os.path.join(ROOT_DIR, 'data/immunotherapy_response/panel_nsclc_pd1_msk_2018/labels_nsclc_pd1_2018.tsv')
        df = pd.read_csv(metadata_path, sep='\t')
        df = df[df['oncotree_code'] == 'LUAD']
        df = df[df['treatment_type'] == 'Monotherapy']
        df = df[df['durable_clinical_benefit'] != 'NE'].values
        labels = []
        for s in new_samples:
            for l in df:
                if s in l[0]:
                    if l[-2] == 'YES':
                        labels.append(1)
                    elif l[-2] == 'NO':
                        labels.append(0)
        data = np.array(labels)
        active_signatures = [1, 2, 4, 5, 6, 13, 17]
    elif 'allen2018' in dataset:
        if 'full' in dataset:
            path = os.path.join(ROOT_DIR, 'data/immunotherapy_response/wxs_mixed_allen_2018/raw/count96-mixed_allen_2018.tsv')
        elif 'panel' in dataset:
            path = os.path.join(ROOT_DIR, 'data/immunotherapy_response/wxs_mixed_allen_2018/processed/deterministic/filter-mixed_allen_2018-WXS_counts.tsv')
        elif 'labels' in dataset:
            path = os.path.join(ROOT_DIR, 'data/immunotherapy_response/panel_nsclc_pd1_msk_2018/labels_nsclc_pd1_2018.tsv')
        elif 'sigma-exp' in dataset:
            path = os.path.join(ROOT_DIR, 'data/nature2019/SigMA_output.tsv')
        else:
            raise ValueError('No such dataset {}'.format(dataset))
        df = pd.read_csv(path, sep='\t')
        data = np.array(df)[:, 1:].astype('float')
        active_signatures = [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]
    else:
        raise ValueError('No such dataset {}'.format(dataset))

    if 'part' in dataset:
        part = dataset.split('part')[-1]
        num_samples = len(data)
        if part == '1':
            data = data[:num_samples // 2]
        elif part == '2':
            data = data[num_samples // 2:]

    active_signatures = np.array(active_signatures) - 1
    return data, active_signatures


def get_model(parameters):
    parameters = {key: np.array(val) for key, val in parameters.items()}
    num_clusters = len(parameters['w'])
    num_topics = len(parameters['e'])
    model = Mix(num_clusters, num_topics, init_params=parameters)
    return model


def get_cosmic_signatures(dir_path=None):
    dir_path = os.path.join(ROOT_DIR, 'data/signatures/COSMIC/cosmic-signatures.tsv') if dir_path is None else dir_path
    return pd.read_csv(dir_path, sep='\t', index_col=0).values


def sigma_output_to_exposures(sigma_output):
    all_df = pd.read_csv(sigma_output, sep='\t')
    # In case this is comma separated
    if len(all_df.columns) == 1:
        all_df = pd.read_csv(sigma_output, sep=',')
    sigs = all_df['sigs_all'].to_list()
    exposures = all_df['exps_all'].to_list()
    output = np.zeros((len(sigs), 30))
    for i, (a, b) in enumerate(zip(sigs, exposures)):
        sig_indices = [int(s.split('_')[1]) - 1 for s in a.split('.')]
        exp_values = np.array([float(s) for s in b.split('_')])
        output[i, sig_indices] = exp_values

    return output


def sigma_output_to_clusters(sigma_output):
    all_df = pd.read_csv(sigma_output, sep='\t')
    # In case this is comma separated
    if len(all_df.columns) == 1:
        all_df = pd.read_csv(sigma_output, sep=',')
    clusters = all_df['categ'].to_list()

    all_clusters = ['Signature_17', 'Signature_3_hc', 'Signature_3_lc', 'Signature_8', 'Signature_APOBEC',
                    'Signature_clock']
    num_clusters = len(all_clusters)
    cluster_to_num = {all_clusters[i]: i for i in range(num_clusters)}
    output = np.zeros((len(clusters), num_clusters))

    for i, c in enumerate(clusters):
        output[i, cluster_to_num[c]] = 1

    return output
