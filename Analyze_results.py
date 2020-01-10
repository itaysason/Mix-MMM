import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_model, load_json, get_data, get_cosmic_signatures
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from scipy.optimize import nnls
from sklearn.metrics.cluster import mutual_info_score, adjusted_mutual_info_score


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
                        print(curr_experiment_string + 'completely missing folds {} and missing runs for {}'.format(
                            no_dir_folds, no_run_folds))
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
            BIC_scores = []
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
                        num_sigs = int(model.split('_')[2][:3])
                        num_clusters = int(model.split('_')[1][:3])
                        num_params = (num_clusters - 1) + (num_sigs - 1) * num_clusters
                        print(model, np.log(num_data_points) * num_params - 2 * best_score, best_score)
                        BIC_scores.append(np.log(num_data_points) * num_params - 2 * best_score)
                        clusters.append(num_clusters)
                num_clusters = np.array(clusters, dtype='int')
                BIC_scores = np.array(BIC_scores)
                print(num_clusters[np.argmin(BIC_scores)])
                plt.plot(num_clusters, BIC_scores)
                plt.title(dataset)
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
                        num_sigs = int(model.split('_')[2][:3])
                        num_clusters = int(model.split('_')[1][:3])
                        num_params = (num_clusters - 1) + (num_sigs - 1) * num_clusters + (96 - 1) * num_sigs
                        # If we use smoothing I think we need to add the number of "data points" we added
                        print(model, np.log(num_data_points) * num_params - 2 * best_score, best_score)
                        clusters.append(num_clusters)
                        sigs.append(num_sigs)
                        BIC_scores.append(np.log(num_data_points) * num_params - 2 * best_score)
                print('\n')
                clusters = np.array(clusters, dtype='int')
                sigs = np.array(sigs, dtype='int')
                BIC_scores = np.array(BIC_scores)
                print(clusters[BIC_scores.argmin()], sigs[BIC_scores.argmin()])

                unique_clusters = np.unique(clusters)
                unique_signaturs = np.unique(sigs)
                from mpl_toolkits.mplot3d import axes3d

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                for c in unique_clusters:
                    tmp = clusters == c
                    curr_sigs = sigs[tmp]
                    curr_clusters = clusters[tmp]
                    curr_BIC_scores = BIC_scores[tmp]
                    arg_sort_curr_sigs = np.argsort(curr_sigs)
                    curr_clusters = np.array([curr_clusters[arg_sort_curr_sigs]])
                    curr_sigs = np.array([curr_sigs[arg_sort_curr_sigs]])
                    curr_BIC_scores = np.array([curr_BIC_scores[arg_sort_curr_sigs]])
                    ax.plot_wireframe(curr_clusters, curr_sigs, curr_BIC_scores, rstride=1, cstride=1)
                for s in unique_signaturs:
                    tmp = sigs == s
                    curr_sigs = sigs[tmp]
                    curr_clusters = clusters[tmp]
                    curr_BIC_scores = BIC_scores[tmp]
                    arg_sort_curr_clusters = np.argsort(curr_clusters)
                    curr_clusters = np.array([curr_clusters[arg_sort_curr_clusters]])
                    curr_sigs = np.array([curr_sigs[arg_sort_curr_clusters]])
                    curr_BIC_scores = np.array([curr_BIC_scores[arg_sort_curr_clusters]])
                    ax.plot_wireframe(curr_clusters, curr_sigs, curr_BIC_scores, rstride=1, cstride=1)

                ax.set_xlabel('clusters')
                ax.set_ylabel('signatures')
                ax.set_zlabel('BIC score')
                # plt.title('BIC score {}'.format(dataset))
                plt.savefig('BIC_{}.pdf'.format(dataset))
                plt.show()


def AMI_score(clusters1, clusters2):
    return adjusted_mutual_info_score(clusters1, clusters2)


def MI_score(clusters1, clusters2):
    return mutual_info_score(clusters1, clusters2)


def MI_score_soft_clustering(clusters1, soft_clusters2):
    num_clusters1 = len(np.unique(clusters1))
    num_clusters2 = soft_clusters2.shape[1]
    num_samples = len(clusters1)
    V_intersection_U = np.zeros((num_clusters1, num_clusters2))
    for i in range(num_samples):
        V_intersection_U[clusters1[i]] += soft_clusters2[i]

    V = np.sum(V_intersection_U, 1)
    U = np.sum(V_intersection_U, 0)
    VU = np.outer(V, U)
    return np.nansum((V_intersection_U / num_samples) * np.log(num_samples * V_intersection_U / VU))


def cosine_similarity(a, b):
    q = np.row_stack((a, b))
    cosine_correlations = np.zeros((len(q), len(q)))
    for i in range(len(q)):
        for j in range(len(q)):
            tmp = np.inner(q[i], q[j]) / (np.sqrt(np.inner(q[i], q[i])) * np.sqrt(np.inner(q[j], q[j])))
            cosine_correlations[i, j] = tmp
            cosine_correlations[j, i] = tmp
    return cosine_correlations


def get_signatures_correlations(a, b, no_repetitions=False, similarity='cosine-similarity'):
    num_sigs = len(a)
    sigs = np.zeros(num_sigs, dtype='int')
    corrs = np.zeros(num_sigs)
    signature_correlations = cosine_similarity(a, b)
    for i in range(num_sigs):
        corr_correlations = signature_correlations[i, num_sigs:]
        sigs[i] = np.argmax(corr_correlations) + 1
        corrs[i] = np.max(corr_correlations).round(3)
    return sigs, corrs


def compute_RE_per_sample(mutations, exposures, signatures):
    reconstructed_mutations = exposures @ signatures
    return compute_RE_from_mutations(reconstructed_mutations, mutations)


def compute_RE_from_mutations(mutations1, mutations2):
    normalized_mutations1 = mutations1 / mutations1.sum(1, keepdims=True)
    normalized_mutations2 = mutations2 / mutations2.sum(1, keepdims=True)
    out = np.zeros(len(mutations1))
    for i in range(len(mutations1)):
        # out[i] = np.linalg.norm(normalized_mutations1[i] - normalized_mutations2[i]) / np.linalg.norm(normalized_mutations1[i])
        # out[i] = np.linalg.norm(normalized_mutations1[i] - normalized_mutations2[i])
        out[i] = np.sum(np.abs(normalized_mutations1[i] - normalized_mutations2[i]))
    return out


def plot_cluster_AMI(range_clusters):
    rich_sample_threshold = 10
    data, active_signatures = get_data('MSK-ALL')
    signatures = get_cosmic_signatures()[active_signatures]
    num_data_points = data.sum()

    nnls_exposures = np.zeros((len(data), len(signatures)))
    for i in range(len(data)):
        nnls_exposures[i] = nnls(signatures.T, data[i])[0]

    num_mutations_per_sample = data.sum(1)

    all_df = pd.read_csv("data/processed/oncotype_counts.txt", sep='\t')
    all_df['Counts'] = all_df['Counts'].astype(int)
    all_df = all_df[all_df['Counts'] > 100]
    cancer_types = np.array(all_df['Oncotree'])

    sample_cancer_assignments = []
    for oc in cancer_types:
        dat_f = "data/processed/%s_counts.npy" % oc
        tmp_data = np.array(np.load(dat_f, allow_pickle=True), dtype=np.float64)
        sample_cancer_assignments.extend([oc] * len(tmp_data))
    sample_cancer_assignments = np.array(sample_cancer_assignments)

    MIX_AMI_scores = np.zeros((2, len(range_clusters), 10))
    MIX_refit_AMI_scores = np.zeros((2, len(range_clusters), 10))
    KMeans_AMI_scores = np.zeros((2, len(range_clusters), 10))
    NNLS_KMeans_AMI_scores = np.zeros((2, len(range_clusters), 10))
    for idx, num_clusters in enumerate(range_clusters):
        # MIX denovo
        d = 'experiments/trained_models/MSK-ALL/denovo'
        best_num_sigs = None
        best_bic_score = np.inf
        for model in os.listdir(d):
            model_num_clusters = int(model.split('_')[1][:3])
            model_num_sigs = int(model.split('_')[2][:3])
            if num_clusters != model_num_clusters:
                continue
            experiment_dir = os.path.join(d, model)
            runs = os.listdir(experiment_dir)
            best_score = -np.inf
            for run in runs:
                total_score = load_json(os.path.join(experiment_dir, run))['log-likelihood']
                if total_score > best_score:
                    best_score = total_score
            if len(runs) > 0:
                num_params = (model_num_clusters - 1) + (model_num_sigs - 1) * model_num_clusters + (
                        96 - 1) * model_num_sigs
                bic_score = np.log(num_data_points) * num_params - 2 * best_score
                if bic_score < best_bic_score:
                    best_bic_score = bic_score
                    best_num_sigs = model_num_sigs

        d = 'experiments/trained_models/MSK-ALL/denovo/mix_{}clusters_{}signatures'.format(
            str(num_clusters).zfill(3), str(best_num_sigs).zfill(3))

        for nr, run in enumerate(os.listdir(d)):
            model = get_model(load_json(os.path.join(d, run))['parameters'])
            MIX_soft_clustering = model.soft_cluster(data)
            sample_cluster_assignment_MIX = np.argmax(MIX_soft_clustering, 1)
            MIX_AMI_scores[0, idx, nr] = AMI_score(sample_cancer_assignments, sample_cluster_assignment_MIX)
            MIX_AMI_scores[1, idx, nr] = AMI_score(sample_cancer_assignments[num_mutations_per_sample >= rich_sample_threshold],
                                                   sample_cluster_assignment_MIX[num_mutations_per_sample >= rich_sample_threshold])

        # MIX refit
        d = 'experiments/trained_models/MSK-ALL/refit/mix_{}clusters_017signatures'.format(str(num_clusters).zfill(3))
        for nr, run in enumerate(os.listdir(d)):
            model = get_model(load_json(os.path.join(d, run))['parameters'])
            MIX_refit_soft_clustering = model.soft_cluster(data)
            sample_cluster_assignment_MIX_refit = np.argmax(MIX_refit_soft_clustering, 1)
            MIX_refit_AMI_scores[0, idx, nr] = AMI_score(sample_cancer_assignments, sample_cluster_assignment_MIX_refit)
            MIX_refit_AMI_scores[1, idx, nr] = AMI_score(sample_cancer_assignments[num_mutations_per_sample >= rich_sample_threshold],
                                                         sample_cluster_assignment_MIX_refit[num_mutations_per_sample >= rich_sample_threshold])

        # KMeans clustering
        for i in range(10):
            cluster_model = KMeans(num_clusters)
            cluster_model.fit(data)
            kmeans_clusters = cluster_model.predict(data)
            KMeans_AMI_scores[0, idx, i] = AMI_score(sample_cancer_assignments, kmeans_clusters)
            KMeans_AMI_scores[1, idx, i] = AMI_score(sample_cancer_assignments[num_mutations_per_sample >= rich_sample_threshold],
                                                     kmeans_clusters[num_mutations_per_sample >= rich_sample_threshold])

        # NNLS + KMeans clustering
        for i in range(10):
            cluster_model = KMeans(num_clusters)
            cluster_model.fit(nnls_exposures)
            nnls_kmeans_clusters = cluster_model.predict(nnls_exposures)
            NNLS_KMeans_AMI_scores[0, idx, i] = AMI_score(sample_cancer_assignments, nnls_kmeans_clusters)
            NNLS_KMeans_AMI_scores[1, idx, i] = AMI_score(sample_cancer_assignments[num_mutations_per_sample >= rich_sample_threshold],
                                                          nnls_kmeans_clusters[num_mutations_per_sample >= rich_sample_threshold])

        print('finished {}'.format(num_clusters))

    np.save('MIX_AMI_scores', MIX_AMI_scores)
    np.save('MIX_refit_AMI_scores', MIX_refit_AMI_scores)
    np.save('KMeans_AMI_scores', KMeans_AMI_scores)
    np.save('NNLS_KMeans_AMI_scores', NNLS_KMeans_AMI_scores)
    # plt.plot(range_clusters, MIX_AMI_scores[:, 0], label='MIX-denovo')
    # plt.plot(range_clusters, MIX_refit_AMI_scores[:, 0], label='MIX-refit')
    # plt.plot(range_clusters, KMeans_AMI_scores[:, 0], label='KMeans')
    # plt.plot(range_clusters, NNLS_KMeans_AMI_scores[:, 0], label='NNLS+KMeans')
    # # plt.title('All samples AMI score')
    # plt.xlabel('clusters')
    # plt.ylabel('AMI')
    # plt.legend(loc='lower right')
    # plt.savefig('AMI_all.pdf')
    # plt.show()
    #
    # plt.plot(range_clusters, MIX_AMI_scores[:, 1], label='MIX-denovo')
    # plt.plot(range_clusters, MIX_refit_AMI_scores[:, 1], label='MIX-refit')
    # plt.plot(range_clusters, KMeans_AMI_scores[:, 1], label='KMeans')
    # plt.plot(range_clusters, NNLS_KMeans_AMI_scores[:, 1], label='NNLS+KMeans')
    # # plt.title('Filtered AMI score')
    # plt.xlabel('clusters')
    # plt.ylabel('AMI')
    # plt.legend(loc='lower right')
    # plt.savefig('AMI_filtered.pdf')
    # plt.show()
    return


def plot_cluster_MI_soft_clustering(range_clusters):
    rich_sample_threshold = 10
    data, active_signatures = get_data('MSK-ALL')
    signatures = get_cosmic_signatures()[active_signatures]
    num_data_points = data.sum()

    nnls_exposures = np.zeros((len(data), len(signatures)))
    for i in range(len(data)):
        nnls_exposures[i] = nnls(signatures.T, data[i])[0]

    num_mutations_per_sample = data.sum(1)

    all_df = pd.read_csv("data/processed/oncotype_counts.txt", sep='\t')
    all_df['Counts'] = all_df['Counts'].astype(int)
    all_df = all_df[all_df['Counts'] > 100]
    cancer_types = np.array(all_df['Oncotree'])

    sample_cancer_assignments = []
    sample_cancer_id_assignments = []
    for i, oc in enumerate(cancer_types):
        dat_f = "data/processed/%s_counts.npy" % oc
        tmp_data = np.array(np.load(dat_f, allow_pickle=True), dtype=np.float64)
        sample_cancer_assignments.extend([oc] * len(tmp_data))
        sample_cancer_id_assignments.extend([i] * len(tmp_data))
    sample_cancer_assignments = np.array(sample_cancer_assignments)
    sample_cancer_id_assignments = np.array(sample_cancer_id_assignments)
    shuffled_indices = np.arange(len(sample_cancer_assignments))

    MIX_MI_scores = np.zeros((2, len(range_clusters), 10))
    MIX_soft_MI_scores = np.zeros((2, len(range_clusters), 10))
    MIX_refit_MI_scores = np.zeros((2, len(range_clusters), 10))
    MIX_refit_soft_MI_scores = np.zeros((2, len(range_clusters), 10))
    KMeans_MI_scores = np.zeros((2, len(range_clusters), 10))
    NNLS_KMeans_MI_scores = np.zeros((2, len(range_clusters), 10))
    for idx, num_clusters in enumerate(range_clusters):
        # MIX denovo
        d = 'experiments/trained_models/MSK-ALL/denovo'
        best_num_sigs = None
        best_bic_score = np.inf
        for model in os.listdir(d):
            model_num_clusters = int(model.split('_')[1][:3])
            model_num_sigs = int(model.split('_')[2][:3])
            if num_clusters != model_num_clusters:
                continue
            experiment_dir = os.path.join(d, model)
            runs = os.listdir(experiment_dir)
            best_score = -np.inf
            for run in runs:
                total_score = load_json(os.path.join(experiment_dir, run))['log-likelihood']
                if total_score > best_score:
                    best_score = total_score
            if len(runs) > 0:
                num_params = (model_num_clusters - 1) + (model_num_sigs - 1) * model_num_clusters + (
                        96 - 1) * model_num_sigs
                bic_score = np.log(num_data_points) * num_params - 2 * best_score
                if bic_score < best_bic_score:
                    best_bic_score = bic_score
                    best_num_sigs = model_num_sigs

        d = 'experiments/trained_models/MSK-ALL/denovo/mix_{}clusters_{}signatures'.format(
            str(num_clusters).zfill(3), str(best_num_sigs).zfill(3))

        # MIX denovo soft clustering
        for nr, run in enumerate(os.listdir(d)):
            model = get_model(load_json(os.path.join(d, run))['parameters'])
            MIX_soft_clustering = model.soft_cluster(data)
            MIX_soft_MI_scores[0, idx, nr] = MI_score_soft_clustering(sample_cancer_id_assignments, MIX_soft_clustering)
            MIX_soft_MI_scores[1, idx, nr] = MI_score_soft_clustering(sample_cancer_id_assignments[num_mutations_per_sample >= rich_sample_threshold],
                                                                      MIX_soft_clustering[num_mutations_per_sample >= rich_sample_threshold])
            sample_cluster_assignment_MIX = np.argmax(MIX_soft_clustering, 1)
            MIX_MI_scores[0, idx, nr] = MI_score(sample_cancer_assignments, sample_cluster_assignment_MIX)
            MIX_MI_scores[1, idx, nr] = MI_score(sample_cancer_assignments[num_mutations_per_sample >= rich_sample_threshold],
                                                 sample_cluster_assignment_MIX[num_mutations_per_sample >= rich_sample_threshold])

        # MIX refit soft clustering
        d = 'experiments/trained_models/MSK-ALL/refit/mix_{}clusters_017signatures'.format(str(num_clusters).zfill(3))
        for nr, run in enumerate(os.listdir(d)):
            model = get_model(load_json(os.path.join(d, run))['parameters'])
            MIX_refit_soft_clustering = model.soft_cluster(data)
            MIX_refit_soft_MI_scores[0, idx, nr] = MI_score_soft_clustering(sample_cancer_id_assignments, MIX_refit_soft_clustering)
            MIX_refit_soft_MI_scores[1, idx, nr] = MI_score_soft_clustering(sample_cancer_id_assignments[num_mutations_per_sample >= rich_sample_threshold],
                                                                            MIX_refit_soft_clustering[num_mutations_per_sample >= rich_sample_threshold])
            sample_cluster_assignment_MIX_refit = np.argmax(MIX_refit_soft_clustering, 1)
            MIX_refit_MI_scores[0, idx, nr] = MI_score(sample_cancer_assignments, sample_cluster_assignment_MIX_refit)
            MIX_refit_MI_scores[1, idx, nr] = MI_score(sample_cancer_assignments[num_mutations_per_sample >= rich_sample_threshold],
                                                       sample_cluster_assignment_MIX_refit[num_mutations_per_sample >= rich_sample_threshold])

        # KMeans clustering
        for i in range(10):
            cluster_model = KMeans(num_clusters)
            # Shuffling before training to make sure scikit doesn't assume anything on the order
            np.random.shuffle(shuffled_indices)
            shuffled_data = data[shuffled_indices]
            cluster_model.fit(shuffled_data)
            kmeans_clusters = cluster_model.predict(data)
            KMeans_MI_scores[0, idx, i] = MI_score(sample_cancer_assignments, kmeans_clusters)
            KMeans_MI_scores[1, idx, i] = MI_score(sample_cancer_assignments[num_mutations_per_sample >= rich_sample_threshold],
                                                      kmeans_clusters[num_mutations_per_sample >= rich_sample_threshold])

        # NNLS + KMeans clustering
        for i in range(10):
            cluster_model = KMeans(num_clusters)
            # Shuffling before training to make sure scikit doesn't assume anything on the order
            np.random.shuffle(shuffled_indices)
            shuffled_nnls_data = nnls_exposures[shuffled_indices]
            cluster_model.fit(shuffled_nnls_data)
            nnls_kmeans_clusters = cluster_model.predict(nnls_exposures)
            NNLS_KMeans_MI_scores[0, idx, i] = MI_score(sample_cancer_assignments, nnls_kmeans_clusters)
            NNLS_KMeans_MI_scores[1, idx, i] = MI_score(sample_cancer_assignments[num_mutations_per_sample >= rich_sample_threshold],
                                                      nnls_kmeans_clusters[num_mutations_per_sample >= rich_sample_threshold])

        print('finished {}'.format(num_clusters))

    np.save('MIX_MI_scores', MIX_MI_scores)
    np.save('MIX_soft_MI_scores2', MIX_soft_MI_scores)
    np.save('MIX_refit_MI_scores', MIX_refit_MI_scores)
    np.save('MIX_refit_soft_MI_scores', MIX_refit_soft_MI_scores)
    np.save('KMeans_MI_scores', KMeans_MI_scores)
    np.save('NNLS_KMeans_MI_scores', NNLS_KMeans_MI_scores)
    return



def plot_sig_correlations(range_signatures):
    cosmic_signatures = get_cosmic_signatures()
    random_seeds = [140296, 142857, 314179, 847662, 3091985, 28021991, 554433, 123456, 654321, 207022]
    mix_dir = 'experiments/trained_models/MSK-ALL/denovo/'
    a, _ = get_data('MSK-ALL')
    num_data_points = a.sum()
    for fig_pos, num_sigs in enumerate(range_signatures):
        x_axis = np.array([str(i + 1) for i in range(num_sigs)])
        plt.axhline(0.80, color='grey', linestyle='--', label='_nolegend_')
        # plt.axhline(0.85, color='grey', linestyle='--', label='_nolegend_')
        # plt.axhline(0.90, color='grey', linestyle='--', label='_nolegend_')
        # plt.axhline(0.95, color='grey', linestyle='--', label='_nolegend_')
        best_model_path = ''
        best_bic_score = np.inf
        for model in os.listdir(mix_dir):
            model_num_clusters = int(model.split('_')[1][:3])
            model_num_sigs = int(model.split('_')[2][:3])
            if num_sigs != model_num_sigs:
                continue
            experiment_dir = os.path.join(mix_dir, model)
            runs = os.listdir(experiment_dir)
            best_score = -np.inf
            best_run_path = ''
            for run in runs:
                total_score = load_json(os.path.join(experiment_dir, run))['log-likelihood']
                if total_score > best_score:
                    best_score = total_score
                    best_run_path = os.path.join(experiment_dir, run)
            if len(runs) > 0:
                num_params = (model_num_clusters - 1) + (model_num_sigs - 1) * model_num_clusters + (
                        96 - 1) * model_num_sigs
                bic_score = np.log(num_data_points) * num_params - 2 * best_score
                if bic_score < best_bic_score:
                    best_bic_score = bic_score
                    best_model_path = best_run_path
        e = np.array(load_json(best_model_path)['parameters']['e'])
        sigs, corrs = get_signatures_correlations(e, cosmic_signatures)
        sigs = sigs[np.argsort(-corrs)]
        corrs = corrs[np.argsort(-corrs)]
        curr_x_axis = x_axis
        # curr_x_axis = x_axis[corrs >= 0.8]
        # sigs = sigs[corrs >= 0.8]
        # corrs = corrs[corrs >= 0.8]
        plt.plot(curr_x_axis, corrs, '.-k', color='C0')
        for i in range(len(sigs)):
            plt.annotate(str(sigs[i]), (i, corrs[i] + 0.002), color='C0')
        print('{} - {} - {} - {}'.format(best_model_path, sigs.tolist(), corrs.tolist(), sum(corrs)))

        for dataset in ['MSK-ALL', 'clustered-MSK-ALL']:
            a, _ = get_data(dataset)
            best_e = np.zeros((num_sigs, 96))
            best_score = np.inf
            for seed in random_seeds:
                # model = LatentDirichletAllocation(num_sigs, random_state=seed)
                model = NMF(num_sigs, random_state=seed)
                # model = NMF(num_sigs, solver='mu', beta_loss=1, max_iter=1000, random_state=seed)
                pi = model.fit_transform(a)
                e = model.components_
                pi *= e.sum(1)
                e /= e.sum(1, keepdims=True)
                score = np.linalg.norm(a - pi @ e)
                if score < best_score:
                    best_score = score
                    best_e = e

            sigs, corrs = get_signatures_correlations(best_e, cosmic_signatures)
            sigs = sigs[np.argsort(-corrs)]
            corrs = corrs[np.argsort(-corrs)]
            curr_x_axis = x_axis
            # curr_x_axis = x_axis[corrs >= 0.8]
            # sigs = sigs[corrs >= 0.8]
            # corrs = corrs[corrs >= 0.8]
            if dataset == 'MSK-ALL':
                color = 'C1'
            else:
                color = 'C2'
            plt.plot(curr_x_axis, corrs, '.-k', color=color)
            for i in range(len(sigs)):
                plt.annotate(str(sigs[i]), (i, corrs[i] + 0.002), color=color)
            print('{} - {} - {}'.format(sigs.tolist(), corrs.tolist(), sum(corrs)))

        plt.yticks(np.arange(6) * 0.2)
        plt.ylabel('Cosine similarity')
        plt.xlabel('Rank of signature')
        # plt.title('{} signatures'.format(num_sigs))
        plt.legend(['MIX', 'NMF', 'clustered-NMF'], loc='lower left')
        plt.savefig('{}-signatures.pdf'.format(num_sigs))
        plt.show()


def RE_BRCA():
    best_experiments = ['experiments/trained_models/BRCA-ds3-part1/refit/mix_001clusters_012signatures',
                        'experiments/trained_models/BRCA-ds3-part2/refit/mix_001clusters_012signatures',
                        'experiments/trained_models/BRCA-ds6-part1/refit/mix_001clusters_012signatures',
                        'experiments/trained_models/BRCA-ds6-part2/refit/mix_001clusters_012signatures',
                        'experiments/trained_models/BRCA-ds9-part1/refit/mix_002clusters_012signatures',
                        'experiments/trained_models/BRCA-ds9-part2/refit/mix_002clusters_012signatures',
                        'experiments/trained_models/BRCA-ds12-part1/refit/mix_002clusters_012signatures',
                        'experiments/trained_models/BRCA-ds12-part2/refit/mix_002clusters_012signatures',
                        'experiments/trained_models/BRCA-ds15-part1/refit/mix_002clusters_012signatures',
                        'experiments/trained_models/BRCA-ds15-part2/refit/mix_002clusters_012signatures',
                        'experiments/trained_models/BRCA-ds18-part1/refit/mix_002clusters_012signatures',
                        'experiments/trained_models/BRCA-ds18-part2/refit/mix_002clusters_012signatures',
                        'experiments/trained_models/BRCA-ds21-part1/refit/mix_003clusters_012signatures',
                        'experiments/trained_models/BRCA-ds21-part2/refit/mix_002clusters_012signatures',
                        'experiments/trained_models/BRCA-ds24-part1/refit/mix_003clusters_012signatures',
                        'experiments/trained_models/BRCA-ds24-part2/refit/mix_002clusters_012signatures',
                        'experiments/trained_models/BRCA-ds27-part1/refit/mix_003clusters_012signatures',
                        'experiments/trained_models/BRCA-ds27-part2/refit/mix_003clusters_012signatures']

    BRCA_data, BRCA_signatures = get_data('ICGC-BRCA')
    num_BRCA_samples = len(BRCA_data)
    signatures = get_cosmic_signatures()[BRCA_signatures]
    tmp = []
    for idx, experiment in enumerate(best_experiments):
        # Find the best model
        best_score = -np.inf
        best_run = None
        for run in os.listdir(experiment):
            curr_score = load_json(os.path.join(experiment, run))['log-likelihood']
            if curr_score >= best_score:
                best_score = curr_score
                best_run = run
        params = load_json(os.path.join(experiment, best_run))['parameters']
        model = get_model(params)

        # Prepare data
        dataset = experiment.split('/')[2]
        ds_size = dataset.split('-')[1][2:]
        if 'part1' in dataset:
            tmp.append([])
            test_dataset = 'BRCA-ds{}-part2'.format(ds_size)
            test_data = BRCA_data[num_BRCA_samples // 2:]
        else:
            test_dataset = 'BRCA-ds{}-part1'.format(ds_size)
            test_data = BRCA_data[:num_BRCA_samples // 2]

        downsampled_test_data, _ = get_data(test_dataset)
        downsampled_train_data, _ = get_data(dataset)
        normalized_test_data = test_data / test_data.sum(1, keepdims=1)

        # MIX RE with cluster's pi
        clusters, _, _ = model.predict(downsampled_test_data)
        exposures = model.pi[clusters]
        cluster_RE = compute_RE_per_sample(normalized_test_data, exposures, signatures)

        # MIX RE with weighted cluster pi
        exposures = model.weighted_exposures(downsampled_test_data)
        weighted_cluster_RE = compute_RE_per_sample(normalized_test_data, exposures, signatures)

        # No model
        no_model_RE = compute_RE_from_mutations(normalized_test_data, downsampled_test_data)

        # NNLS RE
        exposures = np.zeros(exposures.shape)
        for i in range(len(exposures)):
            exposures[i] = nnls(signatures.T, downsampled_test_data[i])[0]
        nnls_RE = compute_RE_per_sample(normalized_test_data, exposures, signatures)

        # Average samples + NNLS
        avg_train = np.sum(downsampled_train_data, 0)
        nnls_avg = nnls(signatures.T, avg_train)[0]
        nnls_avg /= nnls_avg.sum()
        exposures = np.zeros(exposures.shape)
        for i in range(len(exposures)):
            exposures[i] = nnls_avg
        nnls_avg_RE = compute_RE_per_sample(normalized_test_data, exposures, signatures)

        # Average samples
        avg_train = np.sum(downsampled_train_data, 0)
        avg_train /= avg_train.sum()
        predictions = np.zeros(normalized_test_data.shape)
        for i in range(len(predictions)):
            predictions[i] = avg_train
        avg_model_RE = compute_RE_from_mutations(normalized_test_data, predictions)

        # NNLS base-line
        exposures = np.zeros(exposures.shape)
        for i in range(len(exposures)):
            exposures[i] = nnls(signatures.T, normalized_test_data[i])[0]
        nnls_baseline_RE = compute_RE_per_sample(normalized_test_data, exposures, signatures)

        if 'part1' in dataset:
            tmp[-1].append([])
            tmp[-1].append([])
            tmp[-1].append([])
            tmp[-1].append([])
            tmp[-1].append([])
            tmp[-1].append([])
            tmp[-1].append([])
        tmp[-1][0].extend(cluster_RE)
        tmp[-1][1].extend(weighted_cluster_RE)
        tmp[-1][2].extend(no_model_RE)
        tmp[-1][3].extend(nnls_RE)
        tmp[-1][4].extend(nnls_avg_RE)
        tmp[-1][5].extend(avg_model_RE)
        tmp[-1][6].extend(nnls_baseline_RE)

    return tmp


def RE_OV():

    best_experiments = ['experiments/trained_models/OV-ds3-part1/refit/mix_001clusters_003signatures',
                        'experiments/trained_models/OV-ds3-part2/refit/mix_001clusters_003signatures',
                        'experiments/trained_models/OV-ds6-part1/refit/mix_001clusters_003signatures',
                        'experiments/trained_models/OV-ds6-part2/refit/mix_001clusters_003signatures',
                        'experiments/trained_models/OV-ds9-part1/refit/mix_001clusters_003signatures',
                        'experiments/trained_models/OV-ds9-part2/refit/mix_002clusters_003signatures',
                        'experiments/trained_models/OV-ds12-part1/refit/mix_001clusters_003signatures',
                        'experiments/trained_models/OV-ds12-part2/refit/mix_001clusters_003signatures',
                        'experiments/trained_models/OV-ds15-part1/refit/mix_002clusters_003signatures',
                        'experiments/trained_models/OV-ds15-part2/refit/mix_002clusters_003signatures',
                        'experiments/trained_models/OV-ds18-part1/refit/mix_002clusters_003signatures',
                        'experiments/trained_models/OV-ds18-part2/refit/mix_002clusters_003signatures',
                        'experiments/trained_models/OV-ds21-part1/refit/mix_002clusters_003signatures',
                        'experiments/trained_models/OV-ds21-part2/refit/mix_002clusters_003signatures',
                        'experiments/trained_models/OV-ds24-part1/refit/mix_002clusters_003signatures',
                        'experiments/trained_models/OV-ds24-part2/refit/mix_002clusters_003signatures',
                        'experiments/trained_models/OV-ds27-part1/refit/mix_002clusters_003signatures',
                        'experiments/trained_models/OV-ds27-part2/refit/mix_002clusters_003signatures']

    OV_data, OV_signatures = get_data('TCGA-OV')
    num_OV_samples = len(OV_data)
    signatures = get_cosmic_signatures()[OV_signatures]
    tmp = []
    for idx, experiment in enumerate(best_experiments):
        # Find the best model
        best_score = -np.inf
        best_run = None
        for run in os.listdir(experiment):
            curr_score = load_json(os.path.join(experiment, run))['log-likelihood']
            if curr_score >= best_score:
                best_score = curr_score
                best_run = run
        params = load_json(os.path.join(experiment, best_run))['parameters']
        model = get_model(params)

        # Prepare data
        dataset = experiment.split('/')[2]
        ds_size = dataset.split('-')[1][2:]
        if 'part1' in dataset:
            tmp.append([])
            test_dataset = 'OV-ds{}-part2'.format(ds_size)
            test_data = OV_data[num_OV_samples // 2:]
        else:
            test_dataset = 'OV-ds{}-part1'.format(ds_size)
            test_data = OV_data[:num_OV_samples // 2]

        downsampled_test_data, _ = get_data(test_dataset)
        downsampled_train_data, _ = get_data(dataset)
        normalized_test_data = test_data / test_data.sum(1, keepdims=1)

        # MIX RE with cluster's pi
        clusters, _, _ = model.predict(downsampled_test_data)
        exposures = model.pi[clusters]
        cluster_RE = compute_RE_per_sample(normalized_test_data, exposures, signatures)

        # MIX RE with weighted cluster pi
        exposures = model.weighted_exposures(downsampled_test_data)
        weighted_cluster_RE = compute_RE_per_sample(normalized_test_data, exposures, signatures)

        # No model
        no_model_RE = compute_RE_from_mutations(normalized_test_data, downsampled_test_data)

        # NNLS RE
        exposures = np.zeros(exposures.shape)
        for i in range(len(exposures)):
            exposures[i] = nnls(signatures.T, downsampled_test_data[i])[0]
        nnls_RE = compute_RE_per_sample(normalized_test_data, exposures, signatures)

        # Average samples + NNLS
        avg_train = np.sum(downsampled_train_data, 0)
        nnls_avg = nnls(signatures.T, avg_train)[0]
        nnls_avg /= nnls_avg.sum()
        exposures = np.zeros(exposures.shape)
        for i in range(len(exposures)):
            exposures[i] = nnls_avg
        nnls_avg_RE = compute_RE_per_sample(normalized_test_data, exposures, signatures)

        # Average samples
        avg_train = np.sum(downsampled_train_data, 0)
        avg_train /= avg_train.sum()
        predictions = np.zeros(normalized_test_data.shape)
        for i in range(len(predictions)):
            predictions[i] = avg_train
        avg_model_RE = compute_RE_from_mutations(normalized_test_data, predictions)

        # NNLS base-line
        exposures = np.zeros(exposures.shape)
        for i in range(len(exposures)):
            exposures[i] = nnls(signatures.T, normalized_test_data[i])[0]
        nnls_baseline_RE = compute_RE_per_sample(normalized_test_data, exposures, signatures)

        if 'part1' in dataset:
            tmp[-1].append([])
            tmp[-1].append([])
            tmp[-1].append([])
            tmp[-1].append([])
            tmp[-1].append([])
            tmp[-1].append([])
            tmp[-1].append([])
        tmp[-1][0].extend(cluster_RE)
        tmp[-1][1].extend(weighted_cluster_RE)
        tmp[-1][2].extend(no_model_RE)
        tmp[-1][3].extend(nnls_RE)
        tmp[-1][4].extend(nnls_avg_RE)
        tmp[-1][5].extend(avg_model_RE)
        tmp[-1][6].extend(nnls_baseline_RE)

    return tmp


# BRCA_RE = RE_BRCA()
# from scipy.stats import wilcoxon
#
# scores = np.zeros((len(BRCA_RE), len(BRCA_RE[0])))
# p_values = np.zeros((len(BRCA_RE), len(BRCA_RE[0]), len(BRCA_RE[0])))
# for i in range(len(BRCA_RE)):
#     for j1 in range(len(BRCA_RE[i])):
#         scores[i, j1] = sum(BRCA_RE[i][j1]) / len(BRCA_RE[i][j1])
#         for j2 in range(j1 + 1, len(BRCA_RE[i])):
#             p_values[i, j1, j2] = wilcoxon(BRCA_RE[i][j1], BRCA_RE[i][j2])[1]
#
# np.savetxt('BRCA_RE.tsv', scores, delimiter='\t')
# plot_cluster_AMI(range(1, 16))
# # plot_cluster_MI_soft_clustering(range(8, 16))
#
# MIX_AMI_scores = np.load('MIX_AMI_scores.npy')
# # MIX_soft_AMI_scores = np.load('MIX_soft_AMI_scores.npy')
# MIX_refit_AMI_scores = np.load('MIX_refit_AMI_scores.npy')
# # MIX_refit_soft_AMI_scores = np.load('MIX_refit_soft_AMI_scores.npy')
# KMeans_AMI_scores = np.load('KMeans_AMI_scores.npy')
# NNLS_KMeans_AMI_scores = np.load('NNLS_KMeans_AMI_scores.npy')
# for i in range(2):
#     x = range(1, len(MIX_AMI_scores[i]) + 1)
#     plt.errorbar(x, np.mean(MIX_AMI_scores[i], axis=-1), np.std(MIX_AMI_scores[i], axis=-1), label='MIX-denovo', barsabove=True)
#     # plt.errorbar(x, np.mean(MIX_soft_AMI_scores[i], axis=-1), np.std(MIX_AMI_scores[i], axis=-1), label='soft-MIX-denovo', barsabove=True)
#     plt.errorbar(x, np.mean(MIX_refit_AMI_scores[i], axis=-1), np.std(MIX_refit_AMI_scores[i], axis=-1), label='MIX-refit', barsabove=True)
#     # plt.errorbar(x, np.mean(MIX_refit_soft_AMI_scores[i], axis=-1), np.std(MIX_refit_AMI_scores[i], axis=-1), label='soft-MIX-refit', barsabove=True)
#     plt.errorbar(x, np.mean(KMeans_AMI_scores[i], axis=-1), np.std(KMeans_AMI_scores[i], axis=-1), label='KMeans', barsabove=True)
#     plt.errorbar(x, np.mean(NNLS_KMeans_AMI_scores[i], axis=-1), np.std(NNLS_KMeans_AMI_scores[i], axis=-1), label='NNLS+KMeans', barsabove=True)
#     plt.xlabel('clusters')
#     plt.ylabel('AMI')
#     plt.legend(loc='lower right')
#     plt.savefig('AMI_all.pdf')
#     plt.show()


# plot_sig_correlations(range(5, 12))
# process_BIC()
