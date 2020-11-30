import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils import get_model, load_json, get_data, get_cosmic_signatures, sigma_output_to_exposures, sigma_output_to_clusters
from src.models.MMM import MMM
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.metrics.cluster import mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import CCA
from sklearn.cluster import KMeans
from sklearn.svm import OneClassSVM
from scipy.optimize import nnls
from scipy.stats import spearmanr, pearsonr
from src.constants import ROOT_DIR


import warnings
warnings.filterwarnings("ignore")
def_trained_models_dir = os.path.join(ROOT_DIR, 'experiments/trained_models')


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


def Jaccard_score(clusters1, clusters2):
    N11 = 0
    N10 = 0
    N01 = 0
    num_samples = len(clusters1)
    for i in range(num_samples):
        for j in range(i + 1, num_samples):
            if clusters1[i] == clusters1[j]:
                if clusters2[i] == clusters2[j]:
                    N11 += 1
                else:
                    N10 += 1
            elif clusters2[i] == clusters2[j]:
                N01 += 1
    return N11 / (N11 + N10 + N01)


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
        sigs[i] = np.argmax(corr_correlations)
        corrs[i] = np.max(corr_correlations).round(3)
    return sigs, corrs


def compute_RE_per_sample(mutations, exposures, signatures):
    reconstructed_mutations = exposures @ signatures
    return compute_RE_from_mutations(reconstructed_mutations, mutations, signatures)


def compute_RE_from_mutations(mutations1, mutations2, signatures):
    normalized_mutations1 = mutations1 / mutations1.sum(1, keepdims=True)
    normalized_mutations2 = mutations2 / mutations2.sum(1, keepdims=True)
    out = np.zeros(len(mutations1))
    for i in range(len(mutations1)):
        for j in range(mutations1.shape[1]):
            if normalized_mutations1[i, j] == 0:
                continue
            out[i] += normalized_mutations1[i, j] * np.log(normalized_mutations1[i, j])
            if normalized_mutations2[i, j] != 0:
                out[i] -= normalized_mutations1[i, j] * np.log(normalized_mutations2[i, j])
        # out[i] = np.sum(np.abs(normalized_mutations1[i] - normalized_mutations2[i]))
    return out


def compute_exposures_RE_per_sample(mutations, exposures, signatures):
    nnls_exposures = stack_nnls(mutations, signatures)
    return compute_RE_from_mutations(nnls_exposures, exposures, signatures)


def compute_exposures_RE_from_mutations(mutations1, mutations2, signatures):
    nnls_exposures1 = stack_nnls(mutations1, signatures)
    nnls_exposures2 = stack_nnls(mutations2, signatures)
    return compute_RE_from_mutations(nnls_exposures1, nnls_exposures2, signatures)


def stack_nnls(data, signatures):
    exposures = []
    for m in data:
        exposures.append(nnls(signatures.T, m)[0])
    return np.array(exposures)


def mmm_refitting(data, signatures):
    num_signatures = len(signatures)
    mmm = MMM(num_signatures, init_params={'e': signatures})
    mmm.refit(data)
    exposures = mmm.pi
    return exposures


def poison_regression(data, signatures):
    num_signatures = len(signatures)
    mmm = MMM(num_signatures, init_params={'e': signatures})
    mmm.refit(data)
    exposures = mmm.pi * data.sum(1, keepdims=True)
    return exposures


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


def get_features(model_name, data, active_cosmic_sigs=None):
    num_mutations_per_sample = data.sum(1, keepdims=True)

    # Get exposures
    feature_names = []
    sigs = []
    if model_name == 'NUM_MUT':
        exposures = num_mutations_per_sample
        feature_names.append('num_mut')
    elif model_name == 'data':
        exposures = data.copy()
        feature_names = pd.read_csv('data/signatures/COSMIC/cosmic-signatures.tsv', sep='\t', index_col=0).columns
    elif model_name == 'MIX_MSK-ALL_refit' or model_name == 'MIX_MSK-ALL_refit_normalized':
        directory = os.path.join(ROOT_DIR, 'experiments/trained_models/MSK-ALL/refit')
        model = get_best_model(directory, return_model=True)
        _, sigs = get_data('MSK-ALL')
        exposures = model.weighted_exposures(data)
        exposures = exposures[:, [3, 5]]
        exposures[:, 1] = num_mutations_per_sample[:, 0]
        feature_names = ['sig4', '#mutations']
        exposures *= num_mutations_per_sample
    elif model_name == 'MIX_MSK-ALL_denovo' or model_name == 'MIX_MSK-ALL_denovo_normalized':
        directory = os.path.join(ROOT_DIR, 'experiments/trained_models/MSK-ALL/denovo')
        model = get_best_model(directory, return_model=True)
        num_sigs = model.num_topics
        exposures = model.weighted_exposures(data)
        sigs, corrs = get_signatures_correlations(model.e, get_cosmic_signatures())
        feature_names = ['{} ({})'.format(sigs[i], corrs[i]) for i in range(num_sigs)]
        exposures = exposures[:, [4, 5]]
        exposures[:, 1] = num_mutations_per_sample[:, 0]
        feature_names = ['sig4', '#mutations']
        exposures *= num_mutations_per_sample
    elif model_name == 'NNLS_MSK-ALL':
        _, sigs = get_data('MSK-ALL')
        exposures = stack_nnls(data, get_cosmic_signatures()[sigs])
        exposures = exposures[:, [3, 4]]
        exposures[:, 1] = num_mutations_per_sample[:, 0]
        feature_names = ['sig4', '#mutations']
    elif model_name == 'NNLS':
        sigs = active_cosmic_sigs
        exposures = stack_nnls(data, get_cosmic_signatures()[sigs])
        # exposures = mmm_refitting(data, get_cosmic_signatures()[sigs]) * num_mutations_per_sample
    elif model_name == 'MIX_MSK-ALL_cluster-prob' or model_name == 'MIX_MSK-ALL_cluster-prob_normalized':
        directory = os.path.join(ROOT_DIR, 'experiments/trained_models/MSK-ALL/refit')
        model = get_best_model(directory, return_model=True)
        exposures = model.soft_cluster(data)
        feature_names = ['cluster {}'.format(i + 1) for i in range(model.num_clusters)]
        exposures *= num_mutations_per_sample
    elif model_name == 'MIX_MSK-ALL_denovo_cluster-prob' or model_name == 'MIX_MSK-ALL_denovo_cluster-prob_normalized':
        directory = os.path.join(ROOT_DIR, 'experiments/trained_models/MSK-ALL/denovo')
        model = get_best_model(directory, return_model=True)
        exposures = model.soft_cluster(data)
        feature_names = ['cluster {}'.format(i + 1) for i in range(model.num_clusters)]
        exposures *= num_mutations_per_sample
    else:
        raise NotImplementedError('Did not implemet model {}'.format(model_name))

    if 'normalized' in model_name:
        exposures /= num_mutations_per_sample

    if len(feature_names) == 0:
        feature_names = ['{}'.format(sigs[i] + 1) for i in range(len(sigs))]

    return exposures, feature_names


def score_per_feature(features, labels):
    features = features[:, np.newaxis] if len(features.shape) == 1 else features
    num_features = features.shape[1]
    out_dict = {'pearson_corr': [], 'pearson_p_value': [], 'spearman_corr': [], 'spearman_p_value': [],
                'normal_auc': [], 'reverse_auc': [], 'normal_auprc': [], 'reverse_auprc': []}
    for i in range(num_features):
        curr_feature = features[:, i]
        pearson_corr, pearson_p_value = pearsonr(curr_feature, labels)
        spearman_corr, spearman_p_value = spearmanr(curr_feature, labels)
        out_dict['pearson_corr'].append(pearson_corr)
        out_dict['spearman_p_value'].append(spearman_p_value)
        out_dict['spearman_corr'].append(spearman_corr)
        out_dict['pearson_p_value'].append(pearson_p_value)
        out_dict['normal_auc'].append(roc_auc_score(labels, curr_feature))
        out_dict['reverse_auc'].append(roc_auc_score(1 - labels, curr_feature))
        out_dict['normal_auprc'].append(average_precision_score(labels, curr_feature))
        out_dict['reverse_auprc'].append(average_precision_score(1 - labels, curr_feature))

    return out_dict


def score_all_features(features, labels):
    num_samples = len(labels)
    features = features[:, np.newaxis] if len(features.shape) == 1 else features
    out_dict = {}

    # Print cca correlation
    cca = CCA(n_components=1)
    U_c, V_c = cca.fit_transform(features, labels[:, np.newaxis])
    corr, p = pearsonr(U_c.T[0], V_c)
    out_dict['cca'] = corr
    out_dict['cca_p_value'] = p

    # Train classifier and print auc and auprc
    classifier = LogisticRegression()
    classifier.fit(features, labels)
    prediction_probabilities = classifier.predict_proba(features)[:, 1]
    out_dict['auc'] = roc_auc_score(labels, prediction_probabilities)
    out_dict['auprc'] = average_precision_score(labels, prediction_probabilities)

    # Train classifier in LOO
    predictions = labels * 0
    probs = np.zeros(len(labels))
    results = np.zeros((2, 2))
    for i in range(num_samples):
        mask = np.ones(num_samples, dtype=bool)
        mask[i] = 0
        train_data, train_labels = features[mask], labels[mask]
        test_data, test_labels = features[[i]], labels[[i]]
        classifier = LogisticRegression()
        # classifier = SVC()
        classifier.fit(train_data, train_labels)
        predicted_labels = classifier.predict(test_data)
        probs[i] = classifier.predict_proba(test_data)[0, 1]
        predictions[i] = predicted_labels[0]
        results[predictions[i], labels[i]] += 1
    print(results)
    out_dict['auc_loo'] = roc_auc_score(labels, probs)

    # svm = train_auc_maximizing_svm(features, labels, mode='estimated')
    # score = svm.decision_function(features)
    # out_dict['maximized_auc_estimated'] = roc_auc_score(labels, score)
    # svm = train_auc_maximizing_svm(features, labels, mode='exact')
    # score = svm.decision_function(features)
    # out_dict['maximized_auc_exact'] = roc_auc_score(labels, score)
    return out_dict


def train_auc_maximizing_svm(data, labels, mode='estimated'):
    # Genarating z_ij
    positive_samples = data[labels == 1]
    negative_samples = data[labels == 0]
    num_pos = len(positive_samples)
    num_neg = len(negative_samples)
    z = np.zeros((num_pos * num_neg, data.shape[1]))
    for i in range(num_pos):
        for j in range(num_neg):
            z[i * num_neg + j] = positive_samples[i] - negative_samples[j]

    # Creating training data
    if mode == 'exact':
        train_data = z
    elif mode == 'estimated':
        kmeans = KMeans(num_pos + num_neg, init='k-means++')
        kmeans.fit(z)
        train_data = kmeans.cluster_centers_
    else:
        raise ValueError('mode should be estimated or exact')

    # Training SVM
    svm = OneClassSVM(kernel='linear')
    svm.fit(train_data)

    return svm


### Analyzing functions
def process_BIC(trained_models_dir=def_trained_models_dir, plot_title=True, save_plot=False):
    datasets = os.listdir(trained_models_dir)
    for dataset in datasets:
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
    mix_dir = os.path.join(ROOT_DIR, 'experiments/trained_models/MSK-ALL/denovo')
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
                plt.annotate(str(sigs[i] + 1), (i, corrs[i] + 0.002), color=color)
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


def plot_cluster_AMI(range_clusters, computation='AMI'):
    if computation == 'AMI':
        score_func = AMI_score
    elif computation == 'MI':
        score_func = MI_score
    elif computation == 'jaccard':
        score_func = Jaccard_score
    else:
        raise ValueError('{} is not a valid computation'.format(computation))

    rich_sample_threshold = 10
    data, active_signatures = get_data('MSK-ALL')
    signatures = get_cosmic_signatures()[active_signatures]
    num_data_points = data.sum()

    nnls_exposures = np.zeros((len(data), len(signatures)))
    for i in range(len(data)):
        nnls_exposures[i] = nnls(signatures.T, data[i])[0]

    num_mutations_per_sample = data.sum(1)
    rich_samples = num_mutations_per_sample >= rich_sample_threshold

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

    MIX_scores = np.zeros((2, len(range_clusters)))
    MIX_soft_scores = np.zeros((2, len(range_clusters)))
    MIX_refit_scores = np.zeros((2, len(range_clusters)))
    MIX_soft_refit_scores = np.zeros((2, len(range_clusters)))
    KMeans_scores = np.zeros((2, len(range_clusters)))
    NNLS_KMeans_scores = np.zeros((2, len(range_clusters)))
    for idx, num_clusters in enumerate(range_clusters):
        d = os.path.join(ROOT_DIR, 'experiments/trained_models/MSK-ALL/denovo')
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

        d = os.path.join(ROOT_DIR, 'experiments/trained_models/MSK-ALL/denovo/mix_{}clusters_{}signatures'.format(str(num_clusters).zfill(3), str(best_num_sigs).zfill(3)))

        model = get_model(load_json(get_best_run(d))['parameters'])
        MIX_soft_clustering = model.soft_cluster(data)
        sample_cluster_assignment_MIX = np.argmax(MIX_soft_clustering, 1)
        MIX_scores[0, idx] = score_func(sample_cancer_id_assignments, sample_cluster_assignment_MIX)
        MIX_scores[1, idx] = score_func(sample_cancer_id_assignments[rich_samples],
                                            sample_cluster_assignment_MIX[rich_samples])
        if computation == 'MI':
            MIX_soft_scores[0, idx] = MI_score_soft_clustering(sample_cancer_id_assignments, MIX_soft_clustering)
            MIX_soft_scores[1, idx] = MI_score_soft_clustering(sample_cancer_id_assignments[rich_samples],
                                                                   MIX_soft_clustering[rich_samples])

        # MIX refit
        d = os.path.join(ROOT_DIR, 'experiments/trained_models/MSK-ALL/refit/mix_{}clusters_017signatures'.format(str(num_clusters).zfill(3)))
        model = get_model(load_json(get_best_run(d))['parameters'])
        MIX_refit_soft_clustering = model.soft_cluster(data)
        sample_cluster_assignment_MIX_refit = np.argmax(MIX_refit_soft_clustering, 1)
        MIX_refit_scores[0, idx] = score_func(sample_cancer_id_assignments, sample_cluster_assignment_MIX_refit)
        MIX_refit_scores[1, idx] = score_func(sample_cancer_id_assignments[rich_samples],
                                                  sample_cluster_assignment_MIX_refit[rich_samples])
        if computation == 'MI':
            MIX_soft_refit_scores[0, idx] = MI_score_soft_clustering(sample_cancer_id_assignments, MIX_refit_soft_clustering)
            MIX_soft_refit_scores[1, idx] = MI_score_soft_clustering(sample_cancer_id_assignments[rich_samples],
                                                                         MIX_refit_soft_clustering[rich_samples])

        # KMeans clustering
        cluster_model = KMeans(num_clusters, n_init=100)
        np.random.shuffle(shuffled_indices)
        shuffled_data = data[shuffled_indices]
        cluster_model.fit(shuffled_data)
        kmeans_clusters = cluster_model.predict(data)
        KMeans_scores[0, idx] = score_func(sample_cancer_id_assignments, kmeans_clusters)
        KMeans_scores[1, idx] = score_func(sample_cancer_id_assignments[rich_samples],
                                               kmeans_clusters[rich_samples])

        # NNLS + KMeans clustering
        cluster_model = KMeans(num_clusters, n_init=100)
        np.random.shuffle(shuffled_indices)
        shuffled_data = nnls_exposures[shuffled_indices]
        cluster_model.fit(shuffled_data)
        nnls_kmeans_clusters = cluster_model.predict(nnls_exposures)
        NNLS_KMeans_scores[0, idx] = score_func(sample_cancer_id_assignments, nnls_kmeans_clusters)
        NNLS_KMeans_scores[1, idx] = score_func(sample_cancer_id_assignments[rich_samples],
                                                    nnls_kmeans_clusters[rich_samples])

        print('finished {}'.format(num_clusters))

    plt.plot(range_clusters, MIX_scores[0], label='MIX-denovo')
    if computation == 'MI':
        plt.plot(range_clusters, MIX_soft_scores[0], label='MIX-denovo-soft')
    plt.plot(range_clusters, MIX_refit_scores[0], label='MIX-refit')
    if computation == 'MI':
        plt.plot(range_clusters, MIX_soft_refit_scores[0], label='MIX-refit-soft')
    plt.plot(range_clusters, KMeans_scores[0], label='KMeans')
    plt.plot(range_clusters, NNLS_KMeans_scores[0], label='NNLS+KMeans')
    # plt.title('All samples AMI score')
    plt.xlabel('clusters')
    plt.ylabel(computation)
    plt.legend(loc='lower right')
    plt.xticks(np.arange(min(range_clusters), max(range_clusters) + 1, 2))
    plt.savefig('cluster_score_all.pdf')
    plt.show()

    plt.plot(range_clusters, MIX_scores[1], label='MIX-denovo')
    if computation == 'MI':
        plt.plot(range_clusters, MIX_soft_scores[1], label='MIX-denovo-soft')
    plt.plot(range_clusters, MIX_refit_scores[1], label='MIX-refit')
    if computation == 'MI':
        plt.plot(range_clusters, MIX_soft_refit_scores[1], label='MIX-refit-soft')
    plt.plot(range_clusters, KMeans_scores[1], label='KMeans')
    plt.plot(range_clusters, NNLS_KMeans_scores[1], label='NNLS+KMeans')
    # plt.title('Filtered AMI score')
    plt.xlabel('clusters')
    plt.ylabel(computation)
    plt.legend(loc='lower right')
    plt.xticks(np.arange(min(range_clusters), max(range_clusters) + 1, 2))
    plt.savefig('cluster_score_filtered.pdf')
    plt.show()
    return


def RE(dataset, models=None, computation='mutations-RE'):
    # Handle input
    if models is None:
        models = ['MIX-viterbi-clustering', 'MIX-conditional-clustering', 'MIX-soft-clustering', 'NNLS', 'SigMA']

    if dataset == 'BRCA':
        full_dataset_name = 'ICGC-BRCA'
    elif dataset == 'OV':
        full_dataset_name = 'TCGA-OV'
    else:
        raise ValueError('{} is no a valid dataset'.format(dataset))

    if computation == 'mutations-RE':
        RE_func = compute_RE_per_sample
    elif computation == 'exposures-RE':
        RE_func = compute_exposures_RE_per_sample
    else:
        raise ValueError('{} is no a valid computation inpute'.format(computation))

    # Prepare data
    full_data, active_signatures = get_data(full_dataset_name)
    normalized_full_data = full_data / full_data.sum(1, keepdims=1)
    full_panel_data, _ = get_data('{}-panel-full'.format(dataset))
    normalized_full_panel_data = full_panel_data / full_panel_data.sum(1, keepdims=1)

    signatures = get_cosmic_signatures()[active_signatures]

    trained_model_dir = os.path.join(ROOT_DIR, 'experiments/trained_models')
    all_experiments = os.listdir(trained_model_dir)
    results = {s.split('-par')[0]: {} for s in all_experiments if dataset in s and 'part' in s}
    for ds_dataset_name in results:
        results[ds_dataset_name] = {model: [] for model in models}

        # Prepare data
        ds_dataset_part1_name = ds_dataset_name + '-part' + str(1)
        ds_dataset_part2_name = ds_dataset_name + '-part' + str(2)

        ds_data_part1, _ = get_data(ds_dataset_part1_name)
        ds_data_part2, _ = get_data(ds_dataset_part2_name)
        ds_data = np.row_stack((ds_data_part1, ds_data_part2))

        if 'panel' in ds_dataset_name:
            curr_normalized_full_data = normalized_full_panel_data
        else:
            curr_normalized_full_data = normalized_full_data

        # Find the best model
        mix_part1 = get_best_model(os.path.join(trained_model_dir, ds_dataset_part1_name, 'refit'), return_model=True)
        mix_part2 = get_best_model(os.path.join(trained_model_dir, ds_dataset_part2_name, 'refit'), return_model=True)

        # MIX RE with cluster's pi using viterbi
        if 'MIX-viterbi-clustering' in models:
            clusters, _, _ = mix_part1.predict(ds_data_part2)
            exposures_part2 = mix_part1.pi[clusters]
            clusters, _, _ = mix_part2.predict(ds_data_part1)
            exposures_part1 = mix_part2.pi[clusters]
            exposures = np.row_stack((exposures_part1, exposures_part2))
            results[ds_dataset_name]['MIX-viterbi-clustering'] = \
                RE_func(curr_normalized_full_data, exposures, signatures)

        # MIX RE with cluster's pi using conditional probability
        if 'MIX-conditional-clustering' in models:
            clusters = np.argmax(mix_part1.soft_cluster(ds_data_part2), axis=1)
            exposures_part2 = mix_part1.pi[clusters]
            clusters = np.argmax(mix_part2.soft_cluster(ds_data_part1), axis=1)
            exposures_part1 = mix_part2.pi[clusters]
            exposures = np.row_stack((exposures_part1, exposures_part2))
            results[ds_dataset_name]['MIX-conditional-clustering'] = \
                RE_func(curr_normalized_full_data, exposures, signatures)

        # MIX RE with weighted cluster pi
        if 'MIX-soft-clustering' in models:
            exposures_part2 = mix_part1.weighted_exposures(ds_data_part2)
            exposures_part1 = mix_part2.weighted_exposures(ds_data_part1)
            exposures = np.row_stack((exposures_part1, exposures_part2))
            results[ds_dataset_name]['MIX-soft-clustering'] = \
                RE_func(curr_normalized_full_data, exposures, signatures)

        # NNLS RE
        if 'NNLS' in models:
            exposures = []
            for m in ds_data:
                exposures.append(nnls(signatures.T, m)[0])
            exposures = np.array(exposures)
            results[ds_dataset_name]['NNLS'] = RE_func(curr_normalized_full_data, exposures, signatures)

        if 'SigMA' in models:
            if 'panel' in ds_dataset_name:
                path = 'data/sigma_exposure/out-sigma-{}-panel-full.tsv'.format(dataset.lower())
            else:
                ds = ds_dataset_name.split('ds')[-1]
                path = 'data/sigma_exposure/out-sigma-{}-downsize{}.tsv'.format(dataset.lower(), ds.zfill(3))
            # exposures = sigma_output_to_exposures(path)[:, active_signatures]
            # print('Active_signatures: {}'.format(np.where(sigma_output_to_exposures(path).sum(0) > 0)[0] + 1))
            exposures = sigma_output_to_exposures(path)
            exposures /= exposures.sum(1, keepdims=True)
            results[ds_dataset_name]['SigMA'] = RE_func(curr_normalized_full_data, exposures, get_cosmic_signatures())

    # Shallow analysis (no p-values)
    summed_RE_results = {}
    for s in results:
        summed_RE_results[s] = {}
        for m in results[s]:
            summed_RE_results[s][m] = np.sum(results[s][m]) / len(results[s][m])

    df = pd.DataFrame(summed_RE_results).T
    df.to_csv('RE-{}-{}.tsv'.format(dataset, computation), sep='\t')
    return results
