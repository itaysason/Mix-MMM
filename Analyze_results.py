import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_model, load_json, get_data, get_cosmic_signatures, sigma_output_to_exposures
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from sklearn.metrics.cluster import mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from scipy.optimize import nnls
from scipy.stats import spearmanr


import warnings
warnings.filterwarnings("ignore")


### UTILS
def get_best_run(experiment_dir):
    runs = os.listdir(experiment_dir)
    best_score = -np.inf
    if len(runs) == 0:
        return ''
    best_run = runs[0]
    for run in runs:
        total_score = load_json(os.path.join(experiment_dir, run))['log-likelihood']
        if total_score > best_score:
            best_score = total_score
            best_run = run
    return os.path.join(experiment_dir, best_run)


def get_best_model(dataset_dir, return_model=False, return_params=False):
    if len(os.listdir(dataset_dir)) == 0:
        return None
    dataset = os.path.split(os.path.split(dataset_dir)[0])[-1]
    data, _ = get_data(dataset)
    num_data_points = np.sum(data)

    models = []
    BIC_scores = []
    sigs = []
    clusters = []
    for model in os.listdir(dataset_dir):
        experiment_dir = os.path.join(dataset_dir, model)
        best_run = get_best_run(experiment_dir)
        if len(best_run) > 0:
            best_score = load_json(best_run)['log-likelihood']
            num_sigs = int(model.split('_')[2][:3])
            num_clusters = int(model.split('_')[1][:3])
            num_params = (num_clusters - 1) + (num_sigs - 1) * num_clusters + (96 - 1) * num_sigs
            models.append(best_run)
            clusters.append(num_clusters)
            sigs.append(num_sigs)
            BIC_scores.append(np.log(num_data_points) * num_params - 2 * best_score)

    models = np.array(models)
    BIC_scores = np.array(BIC_scores)
    sigs = np.array(sigs, dtype='int')
    clusters = np.array(clusters, dtype='int')
    best_model = models[np.argmin(BIC_scores)]

    if return_model:
        return get_model(load_json(best_model)['parameters'])

    if return_params:
        return {'BIC_scores': BIC_scores, 'num_clusters': clusters, 'model_paths': models, 'num_signatures': sigs}

    return best_model


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
        out[i] = np.sum(np.abs(normalized_mutations1[i] - normalized_mutations2[i]))
    return out


def learn_NMF(data, num_sigs, beta_loss):
    random_seeds = [140296, 142857, 314179, 847662, 3091985, 28021991, 554433, 123456, 654321, 207022]
    num_words = data.shape[1]
    best_e = np.zeros((num_sigs, num_words))
    best_score = np.inf
    for seed in random_seeds:
        if beta_loss == 2:
            model = NMF(num_sigs, random_state=seed)
        else:
            model = NMF(num_sigs, solver='mu', beta_loss=1, random_state=seed)
        pi = model.fit_transform(data)
        e = model.components_
        pi *= e.sum(1)
        e /= e.sum(1, keepdims=True)
        score = np.linalg.norm(data - pi @ e)
        if score < best_score:
            best_score = score
            best_e = e
    return best_e


### Analyzing functions
def process_BIC(trained_models_dir='experiments/trained_models', plot_title=True, save_plot=False):
    datasets = os.listdir(trained_models_dir)
    for dataset in datasets:
        if '2018' not in dataset:
            continue
        dataset_dir = os.path.join(trained_models_dir, dataset)
        for signature_learning in os.listdir(dataset_dir):
            scores_dict = get_best_model(os.path.join(dataset_dir, signature_learning), return_params=True)
            BIC_scores = scores_dict['BIC_scores']
            num_clusters = scores_dict['num_clusters']
            num_signatures = scores_dict['num_signatures']
            model_paths = scores_dict['model_paths']
            print(dataset, signature_learning, model_paths[np.argmin(BIC_scores)])
            if signature_learning == 'refit':
                print(num_clusters[np.argmin(BIC_scores)])
                plt.plot(num_clusters, BIC_scores)
                plt.xlabel('clusters')
                plt.ylabel('BIC')
            elif signature_learning == 'denovo':
                unique_clusters = np.unique(num_clusters)
                unique_signaturs = np.unique(num_signatures)
                from mpl_toolkits.mplot3d import axes3d

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')

                for c in unique_clusters:
                    tmp = num_clusters == c
                    curr_sigs = num_signatures[tmp]
                    curr_clusters = num_clusters[tmp]
                    curr_BIC_scores = BIC_scores[tmp]
                    arg_sort_curr_sigs = np.argsort(curr_sigs)
                    curr_clusters = np.array([curr_clusters[arg_sort_curr_sigs]])
                    curr_sigs = np.array([curr_sigs[arg_sort_curr_sigs]])
                    curr_BIC_scores = np.array([curr_BIC_scores[arg_sort_curr_sigs]])
                    ax.plot_wireframe(curr_clusters, curr_sigs, curr_BIC_scores, rstride=1, cstride=1)
                for s in unique_signaturs:
                    tmp = num_signatures == s
                    curr_sigs = num_signatures[tmp]
                    curr_clusters = num_clusters[tmp]
                    curr_BIC_scores = BIC_scores[tmp]
                    arg_sort_curr_clusters = np.argsort(curr_clusters)
                    curr_clusters = np.array([curr_clusters[arg_sort_curr_clusters]])
                    curr_sigs = np.array([curr_sigs[arg_sort_curr_clusters]])
                    curr_BIC_scores = np.array([curr_BIC_scores[arg_sort_curr_clusters]])
                    ax.plot_wireframe(curr_clusters, curr_sigs, curr_BIC_scores, rstride=1, cstride=1)

                ax.set_xlabel('clusters')
                ax.set_ylabel('signatures')
                ax.set_zlabel('BIC score')
                plt.xticks(unique_clusters)
                plt.yticks(unique_signaturs)

            if plot_title:
                plt.title(dataset)
            if save_plot:
                plt.savefig('BIC_{}-{}.pdf'.format(dataset, signature_learning))
            plt.show()


def plot_sig_correlations(dataset, range_signatures, beta_loss=2, plot_title=True, save_plot=False):
    cosmic_signatures = get_cosmic_signatures()
    mix_dir = 'experiments/trained_models/MSK-ALL/denovo'
    scores_dict = get_best_model(mix_dir, return_params=True)
    BIC_scores = scores_dict['BIC_scores']
    num_signatures = scores_dict['num_signatures']
    model_paths = scores_dict['model_paths']
    for fig_pos, num_sigs in enumerate(range_signatures):
        signatures_dict = {}

        # MIX signatures
        indices = num_signatures == num_sigs
        best_model_path = model_paths[indices][np.argmin(BIC_scores[indices])]
        e = np.array(load_json(best_model_path)['parameters']['e'])
        signatures_dict['MIX'] = e.copy()

        # NMF signatures
        data, _ = get_data(dataset)
        e = learn_NMF(data, num_sigs, beta_loss=beta_loss)
        signatures_dict['NMF'] = e.copy()

        # clustered-NMF signatures (if needed)
        if dataset == 'MSK-ALL':
            data, _ = get_data('clustered-MSK-ALL')
            e = learn_NMF(data, num_sigs, beta_loss=beta_loss)
            signatures_dict['clustered-NMF'] = e.copy()

        plt.rcParams.update({'font.size': 12})
        x_axis = np.array([str(i + 1) for i in range(num_sigs)])
        plt.axhline(0.80, color='grey', linestyle='--', label='_nolegend_')
        legends = []
        for i, model in enumerate(signatures_dict.keys()):
            legends.append(model)
            e = signatures_dict[model]
            sigs, corrs = get_signatures_correlations(e, cosmic_signatures)
            sigs = sigs[np.argsort(-corrs)]
            corrs = corrs[np.argsort(-corrs)]
            curr_x_axis = x_axis
            color = 'C{}'.format(i)
            plt.plot(curr_x_axis, corrs, '.-k', color=color)
            for i in range(len(sigs)):
                plt.annotate(str(sigs[i]), (i, corrs[i] + 0.002), color=color)
            print('{} - {} - {} - {}'.format(model, sigs.tolist(), corrs.tolist(), sum(corrs)))

        plt.yticks(np.arange(2, 6) * 0.2)
        plt.ylabel('Cosine similarity', fontsize='large')
        plt.xlabel('Rank of signature', fontsize='large')
        plt.legend(legends, loc='lower left')
        if plot_title:
            plt.title('{} signatures'.format(num_sigs))
        if save_plot:
            plt.savefig('{}-signatures.pdf'.format(num_sigs))
        plt.show()


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
        print(experiment)
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


def RE_BRCA_panel():
    best_experiments = ['experiments/trained_models/BRCA-panel-part1/refit/mix_002clusters_012signatures',
                        'experiments/trained_models/BRCA-panel-part2/refit/mix_002clusters_012signatures']

    _, BRCA_signatures = get_data('ICGC-BRCA')
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
        if 'part1' in dataset:
            tmp.append([])
            test_data, _ = get_data('BRCA-panel-full-part2')
            downsampled_test_data, _ = get_data('BRCA-panel-part2')
            downsampled_train_data, _ = get_data('BRCA-panel-part1')
        else:
            test_data, _ = get_data('BRCA-panel-full-part1')
            downsampled_test_data, _ = get_data('BRCA-panel-part1')
            downsampled_train_data, _ = get_data('BRCA-panel-part2')

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


def RE_BRCA_panel2():
    best_experiments = ['experiments/trained_models/BRCA-panel/refit/mix_003clusters_012signatures']

    _, BRCA_signatures = get_data('ICGC-BRCA')
    signatures = get_cosmic_signatures()[BRCA_signatures]
    tmp = []
    for idx, experiment in enumerate(best_experiments):
        tmp.append([])
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
        test_data, _ = get_data('BRCA-panel-full')
        downsampled_test_data, _ = get_data('BRCA-panel')
        downsampled_train_data, _ = get_data('BRCA-panel')

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


def RE_OV_panel():
    best_experiments = ['experiments/trained_models/OV-panel-part1/refit/mix_001clusters_003signatures',
                        'experiments/trained_models/OV-panel-part2/refit/mix_002clusters_003signatures']

    _, BRCA_signatures = get_data('TCGA-OV')
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
        if 'part1' in dataset:
            tmp.append([])
            test_data, _ = get_data('OV-panel-full-part2')
            downsampled_test_data, _ = get_data('OV-panel-part2')
            downsampled_train_data, _ = get_data('OV-panel-part1')
        else:
            test_data, _ = get_data('OV-panel-full-part1')
            downsampled_test_data, _ = get_data('OV-panel-part1')
            downsampled_train_data, _ = get_data('OV-panel-part2')

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


def RE_OV_panel2():
    best_experiments = ['experiments/trained_models/OV-panel/refit/mix_002clusters_003signatures']

    _, BRCA_signatures = get_data('TCGA-OV')
    signatures = get_cosmic_signatures()[BRCA_signatures]
    tmp = []
    for idx, experiment in enumerate(best_experiments):
        tmp.append([])
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
        test_data, _ = get_data('OV-panel-full')
        downsampled_test_data, _ = get_data('OV-panel')
        downsampled_train_data, _ = get_data('OV-panel')

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


def correlate_sigs_hrd_nature2019():
    # Find the best model
    directory = 'experiments/trained_models/ICGC-BRCA/refit'
    model = get_best_model(directory, return_model=True)
    num_sigs = model.num_topics

    data, sigs = get_data('nature2019-panel')
    labels, _ = get_data('nature2019-labels')
    labels = labels[:, 0]

    exposures = model.weighted_exposures(data)

    print('\nMIX correlations')
    for i in range(num_sigs):
        curr_sigs = exposures[:, i]
        print(sigs[i] + 1, np.corrcoef(curr_sigs, labels)[0, 1], spearmanr(curr_sigs, labels))
        # plt.plot(labels, curr_sigs, '.')
        # plt.show()

    directory = 'experiments/trained_models/MSK-ALL/refit'
    model = get_best_model(directory, return_model=True)
    num_sigs = model.num_topics

    exposures = model.weighted_exposures(data)
    sigs = [0,  1,  2,  3,  4,  5,  6,  7,  9, 10, 11, 12, 14, 15, 16, 19, 20]

    print('\nMIX correlations')
    for i in range(num_sigs):
        curr_sigs = exposures[:, i]
        print(sigs[i] + 1, np.corrcoef(curr_sigs, labels)[0, 1], spearmanr(curr_sigs, labels))
        # plt.plot(labels, curr_sigs, '.')
        # plt.show()

    directory = 'experiments/trained_models/MSK-ALL/denovo'
    model = get_best_model(directory, return_model=True)
    num_sigs = model.num_topics

    exposures = model.weighted_exposures(data)
    sigs, corrs = [7,  1, 11,  2,  4, 10], [0.98, 0.924, 0.988, 0.83, 0.924, 0.985]

    print('\nMIX correlations')
    for i in range(num_sigs):
        curr_sigs = exposures[:, i]
        print(sigs[i], corrs[i], np.corrcoef(curr_sigs, labels)[0, 1], spearmanr(curr_sigs, labels))
        # plt.plot(labels, curr_sigs, '.')
        # plt.show()

    exposures, _ = get_data('nature2019-sigma-exp')
    col_names = ['Signature_APOBEC_ml', 'Signature_3_ml', 'Signature_8_ml', 'Signature_17_ml', 'Signature_clock_ml',
                 'Signature_3_c', 'exp_sig3', 'Signature_3_l_rat', 'Signature_3_mva']
    print('\nSigMA correlations')
    for i in range(exposures.shape[1]):
        curr_sigs = exposures[:, i]
        print(col_names[i], np.corrcoef(curr_sigs, labels)[0, 1], spearmanr(curr_sigs, labels))


def correlate_sigs_msk2018():
    # Find the best model
    directory = 'experiments/trained_models/msk2018-LUAD/refit'
    model = get_best_model(directory, return_model=True)
    num_sigs = model.num_topics

    data, sigs = get_data('msk2018-LUAD')
    labels, _ = get_data('msk2018-LUAD-labels')

    exposures = model.weighted_exposures(data)

    print('\nMIX correlations')
    for i in range(num_sigs):
        curr_sigs = exposures[:, i]
        print(sigs[i] + 1, np.corrcoef(curr_sigs, labels)[0, 1], spearmanr(curr_sigs, labels))
        # plt.plot(labels, curr_sigs, '.')
        # plt.show()

    directory = 'experiments/trained_models/MSK-ALL/refit'
    model = get_best_model(directory, return_model=True)
    num_sigs = model.num_topics

    exposures = model.weighted_exposures(data)
    sigs = [0,  1,  2,  3,  4,  5,  6,  7,  9, 10, 11, 12, 14, 15, 16, 19, 20]

    print('\nMIX correlations')
    for i in range(num_sigs):
        curr_sigs = exposures[:, i]
        print(sigs[i] + 1, np.corrcoef(curr_sigs, labels)[0, 1], spearmanr(curr_sigs, labels))
        # plt.plot(labels, curr_sigs, '.')
        # plt.show()

    directory = 'experiments/trained_models/MSK-ALL/denovo'
    model = get_best_model(directory, return_model=True)
    num_sigs = model.num_topics

    exposures = model.weighted_exposures(data)
    sigs, corrs = [7,  1, 11,  2,  4, 10], [0.98, 0.924, 0.988, 0.83, 0.924, 0.985]

    print('\nMIX correlations')
    for i in range(num_sigs):
        curr_sigs = exposures[:, i]
        print(sigs[i], corrs[i], np.corrcoef(curr_sigs, labels)[0, 1], spearmanr(curr_sigs, labels))
        # plt.plot(labels, curr_sigs, '.')
        # plt.show()


def predict_hrd():
    for MODEL in ['MIX', 'SigMA_panel']:
        if MODEL == 'MIX':
            exposure_types = ['cluster', 'weighted', 'prediction']
        else:
            exposure_types = ['']

        for MIX_EXPOSURES in exposure_types:
            for NORMALIZED_EXPOSURES in [1, 0]:
                print('{}, {}, {}'.format(MODEL, MIX_EXPOSURES, 'normalized' if NORMALIZED_EXPOSURES else 'not normalized'))
                # Get training exposures
                if MODEL == 'MIX':
                    # directory = 'experiments/trained_models/ICGC-BRCA/refit'
                    directory = 'experiments/trained_models/BRCA-panel/refit'
                    model = get_best_model(directory, return_model=True)

                    # train_mutations, _ = get_data('BRCA-panel-full')
                    train_mutations, _ = get_data('BRCA-panel')

                    if MIX_EXPOSURES == 'weighted':
                        train_data = model.weighted_exposures(train_mutations)
                    elif MIX_EXPOSURES == 'cluster':
                        clusters, _, _ = model.predict(train_mutations)
                        train_data = model.pi[clusters]
                    elif MIX_EXPOSURES == 'prediction':
                        _, topics, _ = model.predict(train_mutations)
                        train_data = topics / topics.sum(1, keepdims=True)
                    else:
                        raise ValueError('No such model {}'.format(MODEL))
                    if not NORMALIZED_EXPOSURES:
                        train_data *= train_mutations.sum(1, keepdims=True)
                elif MODEL == 'SigMA_panel':
                    path = 'data/out-sigma-filter-BRCA-WGS_counts.tsv'
                    train_data = sigma_output_to_exposures(path)
                    if NORMALIZED_EXPOSURES:
                        train_data /= train_data.sum(1, keepdims=True)
                else:
                    raise ValueError('No such model {}'.format(MODEL))

                # Get train labels
                train_labels, _ = get_data('BRCA-panel-hrd')
                train_labels = train_labels[:, 0]

                # Training classifier
                random_forest = RandomForestClassifier()
                params_grid = {'max_depth': [2, 3, 4, 5], 'n_estimators': [1, 4, 8, 10, 20, 30, 40, 50]}
                grid_search = GridSearchCV(random_forest, params_grid, cv=10)
                grid_search.fit(train_data, train_labels)
                # print('Params selected: {}'.format(grid_search.best_params_))
                # print('Score of selected estimator: {:.3f}'.format(grid_search.best_score_))
                random_forest = grid_search.best_estimator_
                random_forest.fit(train_data, train_labels)

                # random_forest = LogisticRegression()
                # random_forest.fit(train_data, train_labels)

                # Get test data
                if MODEL == 'MIX':
                    # directory = 'experiments/trained_models/ICGC-BRCA/refit'
                    directory = 'experiments/trained_models/BRCA-panel/refit'
                    model = get_best_model(directory, return_model=True)

                    test_mutations, _ = get_data('nature2019-panel')

                    if MIX_EXPOSURES == 'weighted':
                        test_data = model.weighted_exposures(test_mutations)
                    elif MIX_EXPOSURES == 'cluster':
                        clusters, _, _ = model.predict(test_mutations)
                        test_data = model.pi[clusters]
                    elif MIX_EXPOSURES == 'prediction':
                        _, topics, _ = model.predict(test_mutations)
                        test_data = topics / topics.sum(1, keepdims=True)
                    else:
                        raise ValueError('No such model {}'.format(MODEL))
                    if not NORMALIZED_EXPOSURES:
                        test_data *= test_mutations.sum(1, keepdims=True)
                elif MODEL == 'SigMA_panel':
                    path = 'data/nature2019/SigMA_output_full.tsv'
                    test_data = sigma_output_to_exposures(path)
                    if NORMALIZED_EXPOSURES:
                        test_data /= test_data.sum(1, keepdims=True)
                else:
                    raise ValueError('No such model {}'.format(MODEL))

                # Get test labels
                test_labels, _ = get_data('nature2019-labels')
                # fixing to 0, 1. Removing intermediate hrd
                test_labels = test_labels[:, 0]
                test_data = test_data[test_labels != 0]
                test_labels = test_labels[test_labels != 0]
                test_labels[test_labels == -1] += 1

                # Test estimator on data
                prediction_probabilities = random_forest.predict_proba(test_data)[:, 1]
                auc_roc = roc_auc_score(test_labels, prediction_probabilities)
                auprc = average_precision_score(test_labels, prediction_probabilities)
                print('auc-roc: {:.3f}, auprc: {:.3f}'.format(auc_roc, auprc))
                # fpr, tpr, thresholds = roc_curve(test_labels, prediction_probabilities)
                # plt.plot(fpr, tpr)
                # plt.title('{}, {}, {}'.format(MODEL, MIX_EXPOSURES, 'normalized' if NORMALIZED_EXPOSURES else 'not normalized'))
                # plt.show()


# predict_hrd()
correlate_sigs_msk2018()
# process_BIC()
# predict_hrd()
# BRCA_RE = RE_OV_panel2()
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
# print(scores)
# np.savetxt('BRCA_RE1.tsv', scores, delimiter='\t')
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
#     plt.plot(x, np.mean(MIX_AMI_scores[i], axis=-1), label='MIX')
#     plt.plot(x, np.mean(MIX_refit_AMI_scores[i], axis=-1), label='MIX-COSMIC')
#     plt.plot(x, np.mean(KMeans_AMI_scores[i], axis=-1), label='KMeans')
#     plt.plot(x, np.mean(NNLS_KMeans_AMI_scores[i], axis=-1), label='NNLS+KMeans')
#     # # plt.errorbar(x, np.mean(MIX_soft_AMI_scores[i], axis=-1), np.std(MIX_AMI_scores[i], axis=-1), label='soft-MIX-denovo', barsabove=True)
#     # plt.errorbar(x, np.mean(MIX_refit_AMI_scores[i], axis=-1), np.std(MIX_refit_AMI_scores[i], axis=-1), label='MIX-refit', barsabove=True)
#     # # plt.errorbar(x, np.mean(MIX_refit_soft_AMI_scores[i], axis=-1), np.std(MIX_refit_AMI_scores[i], axis=-1), label='soft-MIX-refit', barsabove=True)
#     # plt.errorbar(x, np.mean(KMeans_AMI_scores[i], axis=-1), np.std(KMeans_AMI_scores[i], axis=-1), label='KMeans', barsabove=True)
#     # plt.errorbar(x, np.mean(NNLS_KMeans_AMI_scores[i], axis=-1), np.std(NNLS_KMeans_AMI_scores[i], axis=-1), label='NNLS+KMeans', barsabove=True)
#     plt.xlabel('clusters')
#     plt.ylabel('AMI')
#     plt.xticks(x)
#     plt.legend(loc='lower right')
#     extension = 'all' if i == 0 else 'filtered'
#     plt.savefig('AMI_{}.pdf'.format(extension))
#     plt.show()


# plot_sig_correlations('MSK-ALL', range(5, 12))
# process_BIC()
