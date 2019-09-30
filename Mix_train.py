import os
import pandas as pd
from utils import save_json, get_counts
import numpy as np
import sys
from models.Mix import Mix
import time


def sample_cv(num_clusters, random_seed=None, out_dir='experiments/trained_models'):
    data = get_counts()

    random_seed = int(time.time()) if random_seed is None else random_seed
    np.random.seed(random_seed)

    model_name = 'Mix_' + str(num_clusters).zfill(3)
    dataset = 'ICGC-BRCA'

    out_dir_for_file = os.path.join(out_dir, dataset, model_name)

    try:
        os.makedirs(out_dir_for_file)
    except OSError:
        pass

    out_file = out_dir_for_file + "/" + str(random_seed)
    if os.path.isfile(out_file + '.json'):
        print('Experiment with parameters {} {} {} already exist'.format(model_name, dataset, random_seed))
        # return

    scores_dict, parameters = train(num_clusters, data)
    dict_to_save = {'scores': scores_dict, 'parameters': parameters}
    save_json(out_file, dict_to_save)


def train(num_clusters, train_data, epsilon=1e-15, max_iterations=10000):

    cosmic_signatures = pd.read_csv('data/signatures/COSMIC/cosmic-signatures.tsv', sep='\t', index_col=0)
    # cosmic_signatures = cosmic_signatures.values[np.array([1, 2, 3, 5, 6, 8, 13, 17, 18, 20, 26, 30]) - 1]
    cosmic_signatures = cosmic_signatures.values
    model = Mix(num_clusters, len(cosmic_signatures), init_params={'e': cosmic_signatures}, epsilon=epsilon,
                max_iter=max_iterations)
    model.fit(train_data)
    score = model.log_likelihood(train_data)
    parameters = model.get_params()

    parameters['w'] = parameters['w'].tolist()
    parameters['e'] = parameters['e'].tolist()
    parameters['pi'] = parameters['pi'].tolist()

    return score, parameters


if __name__ == '__main__':
    sample_cv(int(sys.argv[1]), int(sys.argv[2]))
