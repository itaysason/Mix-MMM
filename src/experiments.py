from src.models.Mix import Mix
import numpy as np
import time


def split_train_test_sample_cv(data, num_folds, fold, shuffle_seed=None):
    if fold >= num_folds:
        raise ValueError('fold={} but there are total of {} folds'.format(fold, num_folds))

    num_samples = len(data)
    samples = np.arange(num_samples)
    if shuffle_seed is not None:
        np.random.seed(shuffle_seed)
        np.random.shuffle(samples)

    splits = np.array_split(samples, num_folds)
    train_samples = []
    test_samples = []
    for chunk in range(num_folds):
        if chunk == fold:
            test_samples.extend(splits[chunk])
        else:
            train_samples.extend(splits[chunk])
    train_data = data[train_samples]
    test_data = data[test_samples]
    return train_data, test_data


def train_mix(train_data, num_clusters, num_signatures=None, signatures=None, random_seed=None, epsilon=1e-10, max_iter=10000):
    random_seed = time.time() if random_seed is None else random_seed
    np.random.seed(int(random_seed))
    num_signatures = len(signatures) if signatures is not None else num_signatures
    if num_signatures is None:
        raise ValueError('Both signatures and num_signatures are None')
    model = Mix(num_clusters, num_signatures, epsilon=epsilon, max_iter=max_iter)
    if signatures is not None:
        model.e = signatures
        train_ll = model.refit(train_data)
    else:
        train_ll = model.fit(train_data)
    return model, train_ll


def train_test_mix(train_data, test_data, num_clusters, num_signatures=None, signatures=None, random_seed=None, epsilon=1e-10, max_iter=10000):
    model, train_ll = train_mix(train_data, num_clusters, num_signatures, signatures, random_seed, epsilon, max_iter)
    test_ll = model.log_likelihood(test_data)
    return model, train_ll, test_ll
