from src.utils import get_model, load_json, get_data
import os
import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt


def process_sample_cv(sample_cv_dir='experiments/sampleCV'):
    experiment_string = '{}fold CV on {}'

    datasets = os.listdir(sample_cv_dir)
    scores = []
    num_clusters = []
    for dataset in datasets:
        print(dataset + ':')
        dataset_dir = os.path.join(sample_cv_dir, dataset)
        for signature_learning in os.listdir(dataset_dir):
            for model in os.listdir(os.path.join(dataset_dir, signature_learning)):
                experiment_dir = os.path.join(dataset_dir, signature_learning, model)
                shuffle_seed = os.listdir(experiment_dir)[0]
                experiment_dir = os.path.join(experiment_dir, shuffle_seed)
                total_folds = np.array(os.listdir(experiment_dir))
                for num_folds in total_folds:
                    curr_experiment_string = experiment_string.format(num_folds, model)
                    all_folds = [str(i) for i in range(int(num_folds))]
                    cv_dir = os.path.join(experiment_dir, num_folds)
                    folds = os.listdir(cv_dir)
                    no_dir_folds = [fold for fold in all_folds if fold not in folds]
                    no_run_folds = []
                    experiment_score = 0
                    for fold in folds:
                        runs = os.listdir(os.path.join(cv_dir, fold))
                        num_runs = len(runs)
                        if num_runs == 0:
                            no_run_folds.append(fold)
                            continue
                        train_scores = np.zeros(num_runs)
                        test_scores = np.zeros(num_runs)
                        for i, run in enumerate(runs):
                            file_path = os.path.join(cv_dir, fold, run)
                            run_dict = load_json(file_path)
                            train_scores[i] += run_dict['log-likelihood-train']
                            test_scores[i] += run_dict['log-likelihood-test']

                        # Deciding what run to use according to the log likelihood of the train data
                        best_run = np.argmax(train_scores)
                        experiment_score += test_scores[best_run]
                    if len(no_run_folds) > 0 or len(no_dir_folds):
                        print(curr_experiment_string + 'completely missing folds {} and missing runs for {}'.format(no_dir_folds, no_run_folds))
                    else:
                        print(curr_experiment_string + ' score is {}'.format(experiment_score))
                        scores.append(experiment_score)
                        num_clusters.append(model.split('_')[1])
        print('\n')
    num_clusters = np.array(num_clusters, dtype='int')
    scores = np.array(scores)
    print(num_clusters[np.argmax(scores)])
    plt.plot(num_clusters, scores)
    plt.show()


def process_BIC(trained_models_dir='experiments/trained_models'):
    datasets = os.listdir(trained_models_dir)
    for dataset in datasets:
        print(dataset + ':')
        dataset_dir = os.path.join(trained_models_dir, dataset)
        for signature_learning in os.listdir(dataset_dir):
            print(signature_learning)
            tmp = []
            clusters = []
            if signature_learning == 'refit':
                for model in os.listdir(os.path.join(dataset_dir, signature_learning)):
                    experiment_dir = os.path.join(dataset_dir, signature_learning, model)
                    runs = os.listdir(experiment_dir)
                    best_score = -np.inf
                    for run in runs:
                        total_score = load_json(os.path.join(experiment_dir, run))['log-likelihood']
                        if total_score > best_score:
                            best_score = total_score
                    if len(runs) > 0:
                        num_sigs = int(model.split('_')[2])
                        num_clusters = int(model.split('_')[1])
                        num_params = (num_clusters - 1) + (num_sigs - 1) * num_clusters
                        print(model, np.log(3727) * num_params - 2 * best_score, best_score)
                        tmp.append(np.log(3727) * num_params - 2 * best_score)
                        clusters.append(model.split('_')[1])
                print('\n')
                num_clusters = np.array(clusters, dtype='int')
                tmp = np.array(tmp)
                print(num_clusters[np.argmin(tmp)])
                plt.plot(num_clusters, tmp)
                plt.show()
            elif signature_learning == 'denovo':
                sigs = []
                for model in os.listdir(os.path.join(dataset_dir, signature_learning)):
                    experiment_dir = os.path.join(dataset_dir, signature_learning, model)
                    runs = os.listdir(experiment_dir)
                    best_score = -np.inf
                    for run in runs:
                        total_score = load_json(os.path.join(experiment_dir, run))['log-likelihood']
                        if total_score > best_score:
                            best_score = total_score
                    if len(runs) > 0:
                        num_sigs = int(model.split('_')[2])
                        num_clusters = int(model.split('_')[1])
                        num_params = (num_clusters - 1) + (num_sigs - 1) * num_clusters + (96 - 1) * num_sigs
                        print(model, np.log(3727 + 291.84) * num_params - 2 * best_score, best_score)
                        tmp.append(np.log(3727 + 291.84) * num_params - 2 * best_score)
                        clusters.append(model.split('_')[1])
                        sigs.append(num_sigs)
                print('\n')
                clusters = np.array(clusters, dtype='int')
                sigs = np.array(sigs, dtype='int')
                tmp = np.array(tmp)
                print(clusters[tmp.argmin()], sigs[tmp.argmin()])
                clusters = clusters.reshape((6, 6))
                sigs = sigs.reshape((6, 6))
                tmp = tmp.reshape((6, 6))
                from mpl_toolkits.mplot3d import axes3d

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_wireframe(clusters, sigs, tmp, rstride=1, cstride=1)

                plt.show()


def score_trained_model(model_path):
    experiment_dict = load_json(model_path)
    score = experiment_dict['scores']
    parameters = experiment_dict['parameters']
    model = get_model(parameters)
    model.set_data(get_data())
    model.pi = np.log(model.pi)
    model.w = np.log(model.w)
    model.e = np.log(model.e)
    expected_pi_sample_cluster, expected_e_sample_cluster, likelihood_sample_cluster = model.pre_expectation_step()
    likelihood_sample_cluster += model.w
    likelihood_sample_cluster -= logsumexp(likelihood_sample_cluster, 1, keepdims=True)
    best_cluster_likelihood = likelihood_sample_cluster.max(1)
    rest_clusters_likelihood = np.zeros(best_cluster_likelihood.shape)
    for i in range(len(likelihood_sample_cluster)):
        likelihood_sample_cluster[i, np.argmax(likelihood_sample_cluster[i])] = -np.inf
        rest_clusters_likelihood[i] = logsumexp(likelihood_sample_cluster[i])
    return np.sum(best_cluster_likelihood - rest_clusters_likelihood) / len(best_cluster_likelihood), score


def cluster_purity(model_path):
    """
    https://stats.stackexchange.com/questions/95731/how-to-calculate-purity
    :param model_path:
    :return:
    """
    experiment_dict = load_json(model_path)
    parameters = experiment_dict['parameters']
    model = get_model(parameters)
    model.set_data(get_data())
    model.pi = np.log(model.pi)
    model.w = np.log(model.w)
    model.e = np.log(model.e)
    expected_pi_sample_cluster, expected_e_sample_cluster, likelihood_sample_cluster = model.pre_expectation_step()
    likelihood_sample_cluster += model.w
    likelihood_sample_cluster -= logsumexp(likelihood_sample_cluster, 1, keepdims=True)
    best_cluster_likelihood = likelihood_sample_cluster.max(1)
    rest_clusters_likelihood = np.zeros(best_cluster_likelihood.shape)
    for i in range(len(likelihood_sample_cluster)):
        likelihood_sample_cluster[i, np.argmax(likelihood_sample_cluster[i])] = -np.inf
        rest_clusters_likelihood[i] = logsumexp(likelihood_sample_cluster[i])
    return np.sum(best_cluster_likelihood - rest_clusters_likelihood) / len(best_cluster_likelihood)


def predict(model_path):
    experiment_dict = load_json(model_path)
    parameters = experiment_dict['parameters']
    model = get_model(parameters)
    data = get_data()
    return model.predict(data)


def expected_counts(model_path):
    experiment_dict = load_json(model_path)
    parameters = experiment_dict['parameters']
    model = get_model(parameters)
    data = get_data()
    return model.predict(data)


def predict_MMM(model_path):
    experiment_dict = load_json(model_path)
    parameters = experiment_dict['parameters']
    data = get_data()

    pi = np.array(parameters['pi'])
    log_e = np.log(np.array(parameters['e']))

    num_samples = len(data)

    topic_counts = np.zeros((num_samples, len(log_e)), dtype='int')

    for sample in range(num_samples):
        curr_pi = np.log(pi[sample])
        pr_topic_word = (log_e.T + curr_pi).T
        likeliest_topic_per_word = np.argmax(pr_topic_word, axis=0)
        curr_word_counts = data[sample]
        for word in range(len(curr_word_counts)):
            topic_counts[sample, likeliest_topic_per_word[word]] += data[sample, word]

    return topic_counts


def correlate_counts(counts):
    clinical = np.load('data/clinical_no_nans.npy')
    counts = counts[:, counts.sum(0) != 0]
    q = cca_gamma(counts, clinical)
    return q


def cca_gamma(x, y):
    """
    Canonical Correlation Analysis
    Currently only returns the canonical correlations.
    """
    n, p1 = x.shape
    n, p2 = y.shape
    x, y = scale(x), scale(y)

    Qx, Rx = np.linalg.qr(x)
    Qy, Ry = np.linalg.qr(y)

    rankX = np.linalg.matrix_rank(Rx)
    if rankX == 0:
        raise Exception('Rank(X) = 0! Bad Data!')
    elif rankX < p1:
        # warnings.warn("X not full rank!")
        Qx = Qx[:, 0:rankX]
        Rx = Rx[0:rankX, 0:rankX]

    rankY = np.linalg.matrix_rank(Ry)
    if rankY == 0:
        raise Exception('Rank(X) = 0! Bad Data!')
    elif rankY < p2:
        # warnings.warn("Y not full rank!")
        Qy = Qy[:, 0:rankY]
        Ry = Ry[0:rankY, 0:rankY]

    d = min(rankX, rankY)
    svdInput = np.dot(Qx.T, Qy)

    U, r, V = np.linalg.svd(svdInput)
    r = np.clip(r, 0, 1)
    # A = np.linalg.lstsq(Rx, U[:,0:d]) * np.sqrt(n-1)
    # B = np.linalg.lstsq(Ry, V[:,0:d]) * np.sqrt(n-1)

    # TODO: resize A to match inputs

    # return (A,B,r)
    return r


def scale(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


def BIC(d):
    scores = []
    names = []
    for dir_p in os.listdir(d):
        dir_path = os.path.join(d, dir_p)
        max_score = np.log(0)
        for k in os.listdir(dir_path):
            if '_' in k:
                continue
            curr_file = load_json(os.path.join(dir_path, k))
            curr_score = curr_file['scores']
            if max_score < curr_score:
                max_score = curr_score
        curr_k = int(dir_p.split('_')[1])
        names.append(curr_k)
        penalty = np.log(3500000) * (curr_k * 30 - 1)
        bic = penalty - 2 * max_score
        scores.append(bic)

        print(dir_p, len(os.listdir(dir_path)), penalty)

    scores = np.array(scores)
    names = np.array(names)
    i = np.argsort(names)
    names = names[i]
    scores = scores[i]
    plt.plot(names[10:], scores[10:])
    plt.show()
    return


process_BIC()
process_sample_cv()
