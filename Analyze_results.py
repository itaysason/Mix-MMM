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
        data, _ = get_data(dataset)
        num_data_points = np.sum(data)
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
                        print(model, np.log(num_data_points) * num_params - 2 * best_score, best_score)
                        tmp.append(np.log(num_data_points) * num_params - 2 * best_score)
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
                        # If we use smoothing I think we need to add the number of "data points" we added
                        print(model, np.log(num_data_points) * num_params - 2 * best_score, best_score)
                        tmp.append(np.log(num_data_points) * num_params - 2 * best_score)
                        clusters.append(model.split('_')[1])
                        sigs.append(num_sigs)
                print('\n')
                clusters = np.array(clusters, dtype='int')
                sigs = np.array(sigs, dtype='int')
                tmp = np.array(tmp)
                print(clusters[tmp.argmin()], sigs[tmp.argmin()])
                num_clusters_learned, num_sigs_learned = len(np.unique(clusters)), len(np.unique(sigs))
                clusters = clusters.reshape((num_clusters_learned, num_sigs_learned))
                sigs = sigs.reshape((num_clusters_learned, num_sigs_learned))
                tmp = tmp.reshape((num_clusters_learned, num_sigs_learned))
                from mpl_toolkits.mplot3d import axes3d

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_wireframe(clusters, sigs, tmp, rstride=1, cstride=1)

                plt.show()


process_BIC()
# process_sample_cv()
