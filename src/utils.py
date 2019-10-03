import json
import numpy as np
from src.models.Mix import Mix
import pandas as pd


def save_json(file_name, dict_to_save):
    with open(file_name + '.json', 'w') as fp:
        json.dump(dict_to_save, fp)


def load_json(file_name):
    return json.load(open(file_name))


def get_data(dataset):
    if dataset == 'ICGC-BRCA':
        data = np.load('data/BRCA_counts.npy')
        active_signatures = [1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]
    else:
        raise ValueError('{} is not a valid dataset'.format(dataset))
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
