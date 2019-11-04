from src.utils import get_model, load_json, get_data
import os
import numpy as np
import matplotlib.pyplot as plt
from src.utils import load_json, get_data, get_cosmic_signatures
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF


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
                print('\n')
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


def compare_signatures():
    # loading data
    cosmic_signatures = get_cosmic_signatures()

    dirs = ['experiments/trained_models/MSK-ALL/denovo/mix_010clusters_006signatures']
    dirs = ['experiments/trained_models/TCGA-OV/denovo/mix_004clusters_002signatures']
    # dirs = ['experiments/trained_models/OV-ds2/denovo/mix_003clusters_002signatures']
    for d in dirs:
        print('\n{}'.format(d))
        best_score = -np.inf
        best_run = None
        sig_identified = []
        identification_score = []
        for run in os.listdir(d):
            curr_score = load_json(os.path.join(d, run))['log-likelihood']
            if curr_score >= best_score:
                best_score = curr_score
                best_run = run

            e = np.array(load_json(os.path.join(d, run))['parameters']['e'])
            cosmic_signatures = get_cosmic_signatures()
            num_denovo_signatures = len(e)

            signature_correlations = np.corrcoef(e, cosmic_signatures)
            for i in range(num_denovo_signatures):
                corr_correlations = signature_correlations[i, num_denovo_signatures:]
                sig_identified.append(np.argmax(corr_correlations) + 1)
                identification_score.append(np.max(corr_correlations))
            pass

        print('All runs signatures match summary')
        sig_identified = np.array(sig_identified)
        identification_score = np.array(identification_score)
        sorted_indices = np.argsort(-identification_score)
        sig_identified = sig_identified[sorted_indices]
        identification_score = identification_score[sorted_indices]
        print(0.8, len(sig_identified[identification_score > 0.8]),
              np.unique(sig_identified[identification_score > 0.8], return_counts=True))
        print(0.85, len(sig_identified[identification_score > 0.85]),
              np.unique(sig_identified[identification_score > 0.85], return_counts=True))
        print(0.9, len(sig_identified[identification_score > 0.9]),
              np.unique(sig_identified[identification_score > 0.9], return_counts=True))

        print('\nBest run signature matches')
        e = np.array(load_json(os.path.join(d, best_run))['parameters']['e'])
        num_denovo_signatures = len(e)

        signature_correlations = np.corrcoef(e, cosmic_signatures)
        for i in range(num_denovo_signatures):
            corr_correlations = signature_correlations[i, num_denovo_signatures:]
            print(i, np.argmax(corr_correlations) + 1, np.max(corr_correlations))
    return


def compare_signatures_NMF(range_signatures):
    cosmic_signatures = get_cosmic_signatures()
    random_seeds = [140296, 142857, 314179, 847662, 3091985, 28021991, 554433, 123456, 654321, 207022]
    for dataset in ['MSK-ALL', 'clustered-MSK-ALL']:
        a, _ = get_data(dataset)
        for num_sigs in range_signatures:
            print(dataset, num_sigs)
            sigs = np.zeros(num_sigs, dtype='int')
            corrs = np.zeros(num_sigs)
            best_e = np.zeros((num_sigs, 96))
            best_score = np.inf
            for seed in random_seeds:
                model = NMF(num_sigs, max_iter=1000, random_state=seed)
                pi = model.fit_transform(a)
                e = model.components_
                pi *= e.sum(1)
                e /= e.sum(1, keepdims=True)
                score = np.linalg.norm(a - pi @ e)
                if score < best_score:
                    best_score = score
                    best_e = e

            signature_correlations = np.corrcoef(best_e, cosmic_signatures)
            for i in range(num_sigs):
                corr_correlations = signature_correlations[i, num_sigs:]
                sigs[i] = np.argmax(corr_correlations) + 1
                corrs[i] = np.max(corr_correlations).round(3)
                print(i, np.argmax(corr_correlations) + 1, np.max(corr_correlations))
            sigs = sigs[np.argsort(corrs)]
            corrs = corrs[np.argsort(corrs)]
            print('{} - {}'.format(sigs.tolist(), corrs.tolist()))


def cluster_assignment_signature_compatability(model, assignment_intersections, cancer_types):
    cancer_signature_dict = {'LUAD': [1, 2, 4, 5],
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
    cosmic_signatures = get_cosmic_signatures()

    pi = model.pi
    e = model.e
    num_denovo_signatures = len(e)
    signature_correlations = np.corrcoef(e, cosmic_signatures)
    signature_matches = np.zeros(num_denovo_signatures)
    signature_scores = np.zeros(num_denovo_signatures)
    for i in range(num_denovo_signatures):
        corr_correlations = signature_correlations[i, num_denovo_signatures:]
        signature_matches[i] = np.argmax(corr_correlations) + 1
        signature_scores[i] = np.max(corr_correlations)

    for i, cancer in enumerate(cancer_types):
        cancer_active_sigs = cancer_signature_dict[cancer]
        top_cluster = np.argmax(assignment_intersections[:, i])
        top_cluster_score = assignment_intersections[top_cluster, i] / np.sum(assignment_intersections[:, i])
        topc_cluster_active_sigs = signature_matches[pi[top_cluster] > 1e-1]
        print(cancer, top_cluster + 1, top_cluster_score, cancer_active_sigs, topc_cluster_active_sigs)
    return


def AMI_score(clusters1, clusters2):
    from sklearn.metrics.cluster import adjusted_mutual_info_score
    return adjusted_mutual_info_score(clusters1, clusters2)


def compare_clusters():
    # loading data
    data, active_signatures = get_data('MSK-ALL')

    rich_sample_threshold = 10
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

    d = 'experiments/trained_models/MSK-ALL/denovo/mix_010clusters_006signatures'
    best_score = -np.inf
    best_run = None
    for run in os.listdir(d):
        curr_score = load_json(os.path.join(d, run))['log-likelihood']
        if curr_score >= best_score:
            best_score = curr_score
            best_run = run

    model = get_model(load_json(os.path.join(d, best_run))['parameters'])
    sample_cluster_assignment_MIX, _, _ = model.predict(data)
    assignment_intersections = np.zeros((model.num_clusters, len(cancer_types)), dtype='int')
    for cluster in range(model.num_clusters):
        for cancer_idx, cancer in enumerate(cancer_types):
            assignment_intersections[cluster, cancer_idx] += np.count_nonzero(
                np.logical_and(sample_cluster_assignment_MIX == cluster, sample_cancer_assignments == cancer))

    print('MIX clustering on all samples: jaccard = {}'.format(AMI_score(sample_cancer_assignments, sample_cluster_assignment_MIX)))

    assignment_intersections = np.zeros((model.num_clusters, len(cancer_types)), dtype='int')
    for cluster in range(model.num_clusters):
        for cancer_idx, cancer in enumerate(cancer_types):
            assignment_intersections[cluster, cancer_idx] += np.count_nonzero(
                np.logical_and(num_mutations_per_sample >= rich_sample_threshold,
                               np.logical_and(sample_cluster_assignment_MIX == cluster,
                                              sample_cancer_assignments == cancer)))

    print('MIX clustering on samples with {} or more mutations: jaccard: {}'.format(rich_sample_threshold, AMI_score(sample_cancer_assignments[num_mutations_per_sample >= rich_sample_threshold], sample_cluster_assignment_MIX[num_mutations_per_sample >= rich_sample_threshold])))

    # KMeans clustering
    tmp1 = []
    tmp2 = []
    tmp3 = []
    tmp4 = []
    for num_clusters in range(100, 101):
        print(num_clusters)
        model.num_clusters = num_clusters
        cluster_model = KMeans(model.num_clusters)
        cluster_model.fit(data)
        kmeans_clusters = cluster_model.predict(data)
        assignment_intersections = np.zeros((model.num_clusters, len(cancer_types)), dtype='int')
        for cluster in range(model.num_clusters):
            for cancer_idx, cancer in enumerate(cancer_types):
                assignment_intersections[cluster, cancer_idx] += np.count_nonzero(
                    np.logical_and(kmeans_clusters == cluster, sample_cancer_assignments == cancer))

        print('KMeans clustering on all samples: jaccard = {}'.format(AMI_score(sample_cancer_assignments, kmeans_clusters)))
        tmp1.append(AMI_score(sample_cancer_assignments, kmeans_clusters))

        rich_sample_threshold = 10
        assignment_intersections = np.zeros((model.num_clusters, len(cancer_types)), dtype='int')
        for cluster in range(model.num_clusters):
            for cancer_idx, cancer in enumerate(cancer_types):
                assignment_intersections[cluster, cancer_idx] += np.count_nonzero(
                    np.logical_and(num_mutations_per_sample >= rich_sample_threshold,
                                   np.logical_and(kmeans_clusters == cluster, sample_cancer_assignments == cancer)))

        print('KMeans clustering on samples with {} or more mutations: jaccard: {}'.format(rich_sample_threshold, AMI_score(sample_cancer_assignments[num_mutations_per_sample >= rich_sample_threshold], kmeans_clusters[num_mutations_per_sample >= rich_sample_threshold])))
        tmp2.append(AMI_score(sample_cancer_assignments[num_mutations_per_sample >= rich_sample_threshold], kmeans_clusters[num_mutations_per_sample >= rich_sample_threshold]))

        # NMF + KMeans clustering
        d = 'experiments/NMF_denovo_models'
        best_score = -np.inf
        best_run = None
        for run in os.listdir(d):
            curr_score = load_json(os.path.join(d, run))['log-likelihood']
            if curr_score >= best_score:
                best_score = curr_score
                best_run = run

        pi = np.array(load_json(os.path.join(d, best_run))['parameters']['pi'])
        nmf_cluster_model = KMeans(model.num_clusters)
        nmf_cluster_model.fit(pi)
        nmf_kmeans_clusters = nmf_cluster_model.predict(pi)
        for cluster in range(model.num_clusters):
            for cancer_idx, cancer in enumerate(cancer_types):
                assignment_intersections[cluster, cancer_idx] += np.count_nonzero(
                    np.logical_and(nmf_kmeans_clusters == cluster, sample_cancer_assignments == cancer))

        print('NMF + KMeans clustering on all samples: jaccard = {}'.format(
            AMI_score(sample_cancer_assignments, nmf_kmeans_clusters)))
        tmp3.append(AMI_score(sample_cancer_assignments, nmf_kmeans_clusters))

        rich_sample_threshold = 10
        assignment_intersections = np.zeros((model.num_clusters, len(cancer_types)), dtype='int')
        for cluster in range(model.num_clusters):
            for cancer_idx, cancer in enumerate(cancer_types):
                assignment_intersections[cluster, cancer_idx] += np.count_nonzero(
                    np.logical_and(num_mutations_per_sample >= rich_sample_threshold,
                                   np.logical_and(nmf_kmeans_clusters == cluster, sample_cancer_assignments == cancer)))

        print('NMF+ KMeans clustering on samples with {} or more mutations: jaccard: {}'.format(rich_sample_threshold,
                                                                                                AMI_score(
                                                                                               sample_cancer_assignments[
                                                                                                   num_mutations_per_sample >= rich_sample_threshold],
                                                                                               nmf_kmeans_clusters[
                                                                                                   num_mutations_per_sample >= rich_sample_threshold])))
        tmp4.append(AMI_score(sample_cancer_assignments[num_mutations_per_sample >= rich_sample_threshold], nmf_kmeans_clusters[num_mutations_per_sample >= rich_sample_threshold]))
    print(tmp1)
    print(tmp2)
    print(tmp3)
    print(tmp4)
    print(5 + np.argmax(tmp1), np.max(tmp1))
    print(5 + np.argmax(tmp2), np.max(tmp2))
    print(5 + np.argmax(tmp3), np.max(tmp3))
    print(5 + np.argmax(tmp4), np.max(tmp4))


def plot_cluster_AMI(range_clusters):
    MIX_num_sginatures = 6
    rich_sample_threshold = 10
    data, active_signatures = get_data('MSK-ALL')

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

    d = 'experiments/NMF_denovo_models'
    best_score = -np.inf
    best_run = None
    for run in os.listdir(d):
        curr_score = load_json(os.path.join(d, run))['log-likelihood']
        if curr_score >= best_score:
            best_score = curr_score
            best_run = run

    NMF_pi = np.array(load_json(os.path.join(d, best_run))['parameters']['pi'])

    MIX_AMI_scores = np.zeros((len(range_clusters), 2))
    MIX_refit_AMI_scores = np.zeros((len(range_clusters), 2))
    KMeans_AMI_scores = np.zeros((len(range_clusters), 2))
    NMF_KMeans_AMI_scores = np.zeros((len(range_clusters), 2))
    for idx, num_clusters in enumerate(range_clusters):
        # MIX denovo
        d = 'experiments/trained_models/MSK-ALL/denovo/mix_{}clusters_{}signatures'.format(str(num_clusters).zfill(3), str(MIX_num_sginatures).zfill(3))
        best_score = -np.inf
        best_run = None
        for run in os.listdir(d):
            curr_score = load_json(os.path.join(d, run))['log-likelihood']
            if curr_score >= best_score:
                best_score = curr_score
                best_run = run

        model = get_model(load_json(os.path.join(d, best_run))['parameters'])
        sample_cluster_assignment_MIX, _, _ = model.predict(data)
        MIX_AMI_scores[idx, 0] = AMI_score(sample_cancer_assignments, sample_cluster_assignment_MIX)
        MIX_AMI_scores[idx, 1] = AMI_score(sample_cancer_assignments[num_mutations_per_sample >= rich_sample_threshold], sample_cluster_assignment_MIX[num_mutations_per_sample >= rich_sample_threshold])

        # MIX refit
        d = 'experiments/trained_models/MSK-ALL/refit/mix_{}clusters_017signatures'.format(str(num_clusters).zfill(3))
        best_score = -np.inf
        best_run = None
        for run in os.listdir(d):
            curr_score = load_json(os.path.join(d, run))['log-likelihood']
            if curr_score >= best_score:
                best_score = curr_score
                best_run = run

        model = get_model(load_json(os.path.join(d, best_run))['parameters'])
        sample_cluster_assignment_MIX, _, _ = model.predict(data)
        MIX_refit_AMI_scores[idx, 0] = AMI_score(sample_cancer_assignments, sample_cluster_assignment_MIX)
        MIX_refit_AMI_scores[idx, 1] = AMI_score(sample_cancer_assignments[num_mutations_per_sample >= rich_sample_threshold], sample_cluster_assignment_MIX[num_mutations_per_sample >= rich_sample_threshold])

        # KMeans clustering
        nmf_cluster_model = KMeans(num_clusters)
        nmf_cluster_model.fit(data)
        kmeans_clusters = nmf_cluster_model.predict(data)

        KMeans_AMI_scores[idx, 0] = AMI_score(sample_cancer_assignments, kmeans_clusters)
        KMeans_AMI_scores[idx, 1] = AMI_score(sample_cancer_assignments[num_mutations_per_sample >= rich_sample_threshold], kmeans_clusters[num_mutations_per_sample >= rich_sample_threshold])

        # NMF + KMeans clustering
        nmf_cluster_model = KMeans(num_clusters)
        nmf_cluster_model.fit(NMF_pi)
        nmf_kmeans_clusters = nmf_cluster_model.predict(NMF_pi)

        NMF_KMeans_AMI_scores[idx, 0] = AMI_score(sample_cancer_assignments, nmf_kmeans_clusters)
        NMF_KMeans_AMI_scores[idx, 1] = AMI_score(sample_cancer_assignments[num_mutations_per_sample >= rich_sample_threshold], nmf_kmeans_clusters[num_mutations_per_sample >= rich_sample_threshold])

    print(MIX_AMI_scores)
    print(MIX_refit_AMI_scores)
    print(KMeans_AMI_scores)
    print(NMF_KMeans_AMI_scores)
    plt.plot(range_clusters, MIX_AMI_scores[:, 0], label='MIX-denovo')
    plt.plot(range_clusters, MIX_refit_AMI_scores[:, 0], label='MIX-refit')
    plt.plot(range_clusters, KMeans_AMI_scores[:, 0], label='KMeans')
    plt.plot(range_clusters, NMF_KMeans_AMI_scores[:, 0], label='NMF+KMeans')
    # plt.title('All samples AMI score')
    plt.xlabel('clusters')
    plt.ylabel('AMI')
    plt.legend(loc='lower right')
    plt.savefig('AMI_all.pdf')
    plt.show()

    plt.plot(range_clusters, MIX_AMI_scores[:, 1], label='MIX')
    plt.plot(range_clusters, MIX_refit_AMI_scores[:, 1], label='MIX-refit')
    plt.plot(range_clusters, KMeans_AMI_scores[:, 1], label='KMeans')
    plt.plot(range_clusters, NMF_KMeans_AMI_scores[:, 1], label='NMF+KMeans')
    # plt.title('Filtered AMI score')
    plt.xlabel('clusters')
    plt.ylabel('AMI')
    plt.legend(loc='lower right')
    plt.savefig('AMI_filtered.pdf')
    plt.show()
    return


def plot_sig_correlations(range_signatures):
    cosmic_signatures = get_cosmic_signatures()
    random_seeds = [140296, 142857, 314179, 847662, 3091985, 28021991, 554433, 123456, 654321, 207022]
    mix_dir = 'experiments/trained_models/MSK-ALL/denovo/'
    a, _ = get_data('MSK-ALL')
    num_data_points = a.sum()
    for fig_pos, num_sigs in enumerate(range_signatures):
        x_axis = np.array([str(i + 1) for i in range(num_sigs)])
        plt.axhline(0.85, color='grey', linestyle='--', label='_nolegend_')
        plt.axhline(0.9, color='grey', linestyle='--', label='_nolegend_')
        plt.axhline(0.95, color='grey', linestyle='--', label='_nolegend_')
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
                num_params = (model_num_clusters - 1) + (model_num_sigs - 1) * model_num_clusters + (96 - 1) * model_num_sigs
                bic_score = np.log(num_data_points) * num_params - 2 * best_score
                if bic_score < best_bic_score:
                    best_bic_score = bic_score
                    best_model_path = best_run_path
        e = np.array(load_json(best_model_path)['parameters']['e'])
        sigs = np.zeros(num_sigs, dtype='int')
        corrs = np.zeros(num_sigs)
        signature_correlations = np.corrcoef(e, cosmic_signatures)
        for i in range(num_sigs):
            corr_correlations = signature_correlations[i, num_sigs:]
            sigs[i] = np.argmax(corr_correlations) + 1
            corrs[i] = np.max(corr_correlations).round(3)
            signature_correlations[:, num_sigs + np.argmax(corr_correlations)] = 0
        sigs = sigs[np.argsort(-corrs)]
        corrs = corrs[np.argsort(-corrs)]
        curr_x_axis = x_axis[corrs >= 0.8]
        sigs = sigs[corrs >= 0.8]
        corrs = corrs[corrs >= 0.8]
        plt.plot(curr_x_axis, corrs, '.-k', color='C0')
        for i in range(len(sigs)):
            plt.annotate(str(sigs[i]), (i, corrs[i] + 0.002), color='C0')
        print('{} - {} - {} - {}'.format(best_model_path, sigs.tolist(), corrs.tolist(), sum(corrs)))

        for dataset in ['MSK-ALL', 'clustered-MSK-ALL']:
            a, _ = get_data(dataset)
            sigs = np.zeros(num_sigs, dtype='int')
            corrs = np.zeros(num_sigs)
            best_e = np.zeros((num_sigs, 96))
            best_score = np.inf
            for seed in random_seeds:
                model = NMF(num_sigs, max_iter=1000, random_state=seed)
                pi = model.fit_transform(a)
                e = model.components_
                pi *= e.sum(1)
                e /= e.sum(1, keepdims=True)
                score = np.linalg.norm(a - pi @ e)
                if score < best_score:
                    best_score = score
                    best_e = e

            signature_correlations = np.corrcoef(best_e, cosmic_signatures)
            for i in range(num_sigs):
                corr_correlations = signature_correlations[i, num_sigs:]
                sigs[i] = np.argmax(corr_correlations) + 1
                corrs[i] = np.max(corr_correlations).round(3)
                signature_correlations[:, num_sigs + np.argmax(corr_correlations)] = 0
            sigs = sigs[np.argsort(-corrs)]
            corrs = corrs[np.argsort(-corrs)]
            curr_x_axis = x_axis[corrs >= 0.8]
            sigs = sigs[corrs >= 0.8]
            corrs = corrs[corrs >= 0.8]
            if dataset == 'MSK-ALL':
                color = 'C1'
            else:
                color = 'C2'
            plt.plot(curr_x_axis, corrs, '.-k', color=color)
            for i in range(len(sigs)):
                plt.annotate(str(sigs[i]), (i, corrs[i] + 0.002), color=color)
            print('{} - {} - {}'.format(sigs.tolist(), corrs.tolist(), sum(corrs)))

        plt.yticks(0.8 + np.arange(5) * 0.05)
        # plt.title('{} signatures'.format(num_sigs))
        plt.legend(['MIX', 'NMF', 'clustered-NMF'], loc='lower left')
        plt.savefig('{}-signatures.pdf'.format(num_sigs))
        plt.show()


# process_BIC('experiments/trained_models')
# compare_signatures()
# plot_sig_correlations(range(4, 12))
# compare_clusters()
# plot_cluster_AMI(range(1, 15))
