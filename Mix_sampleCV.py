import os
import pandas as pd
from utils import save_json, get_counts
import numpy as np
import sys
from models.Mix import Mix

np.random.seed(1137)


def sample_cv(num_clusters, num_folds, fold, out_dir='experiments/sampleCV'):
    data = get_counts()

    if not 0 <= fold < num_folds:
        raise ValueError('num_folds is {} but fold is {}'.format(num_folds, fold))

    model_name = 'Mix_' + str(num_clusters).zfill(3)
    dataset = 'ICGC-BRCA'

    out_dir_for_file = os.path.join(out_dir, dataset, model_name)

    try:
        os.makedirs(out_dir_for_file)
    except OSError:
        pass

    out_file = out_dir_for_file + "/" + str(fold + 1) + '_' + str(num_folds)
    if os.path.isfile(out_file + '.json'):
        print('Experiment with parameters {} {} {} {} already exist'.format(model_name, dataset, num_folds, fold))
        return

    # splitting the data
    sample_names = np.arange(len(data))
    splits = np.array_split(sample_names, num_folds)
    train_data = []
    test_data = []
    for chunk in range(num_folds):
        if chunk == fold:
            test_data.extend(splits[chunk])
        else:
            train_data.extend(splits[chunk])
    train_data = data[train_data]
    test_data = data[test_data]
    scores_dict, parameters = train_and_test(num_clusters, train_data, test_data)
    dict_to_save = {'scores': scores_dict, 'parameters': parameters}
    save_json(out_file, dict_to_save)


def train_and_test(num_clusters, train_data, test_data, epsilon=1e-15, max_iterations=10000):

    cosmic_signatures = pd.read_csv('data/signatures/COSMIC/cosmic-signatures.tsv', sep='\t', index_col=0)
    # cosmic_signatures = cosmic_signatures.values[np.array([1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]) - 1]
    cosmic_signatures = cosmic_signatures.values
    model = Mix(num_clusters, len(cosmic_signatures), init_params={'e': cosmic_signatures}, epsilon=epsilon,
                max_iter=max_iterations)
    model.fit(train_data)
    score = {'trainScore': model.log_likelihood(train_data), 'testScore': model.log_likelihood(test_data)}
    parameters = model.get_params()

    parameters['w'] = parameters['w'].tolist()
    parameters['e'] = parameters['e'].tolist()
    parameters['pi'] = parameters['pi'].tolist()

    return score, parameters


if __name__ == '__main__':
    sample_cv(int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]))
