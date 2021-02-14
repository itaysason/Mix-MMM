import click
import os
from src.constants import ROOT_DIR
import numpy as np


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(chain=False, context_settings=CONTEXT_SETTINGS)
@click.option('--debug/--no-debug', default=False, help="Default is --no-debug.")
@click.option('-v', '--verbosity', default=0, help='Verbosity level (0-3).')
def simple_cli(debug, verbosity):
    return


@simple_cli.command('reconstruction_error', short_help='Train and save model')
@click.option('--dataset', type=click.Choice(['BRCA', 'OV']))
@click.option('--error', type=click.Choice(['mutations', 'exposures']))
@click.option('--out_dir', type=str, default='results/RE')
def reconstruction_error(dataset, error, out_dir):
    out_dir = os.path.join(ROOT_DIR, out_dir)
    try:
        os.makedirs(out_dir)
    except OSError:
        pass

    from src.analyze_results import RE

    results = RE(dataset, computation=error)
    results.to_csv(os.path.join(out_dir, '{}_{}.tsv').format(dataset, error), sep='\t')


@simple_cli.command('AMI', short_help='Train and save model')
def AMI():
    try:
        os.makedirs(os.path.join(ROOT_DIR, 'results', 'AMI'))
    except OSError:
        pass

    from src.analyze_results import plot_cluster_AMI

    plot_cluster_AMI(range(1, 21))


@simple_cli.command('BIC', short_help='Train and save model')
def BIC():
    try:
        os.makedirs(os.path.join(ROOT_DIR, 'results', 'BIC'))
    except OSError:
        pass

    from src.analyze_results import process_BIC

    process_BIC('MSK-ALL')


@simple_cli.command('signatures_similarity', short_help='Train and save model')
@click.option('--num_sigs', type=int)
def signatures_similarity(num_sigs):
    try:
        os.makedirs(os.path.join(ROOT_DIR, 'results', 'signatures_similarity'))
    except OSError:
        pass

    from src.analyze_results import plot_sig_correlations

    plot_sig_correlations('MSK-ALL', num_sigs)


@simple_cli.command('analyze_synthetic', short_help='Train and save model')
@click.option('--dataset', type=str)
def analyze_synthetic(dataset):

    if 'simulated' not in dataset:
        raise ValueError('dataset is no synthetic')

    try:
        os.makedirs(os.path.join(ROOT_DIR, 'results', 'synthetic', dataset))
    except OSError:
        pass

    from src.analyze_results import simulated_data_analysis

    simulated_data_analysis(dataset)


@simple_cli.command('ROC_HRD', short_help='Train and save model')
def ROC_HRD():

    from src.analyze_results import get_best_model, stack_nnls
    import matplotlib.pyplot as plt
    from src.utils import get_data, get_cosmic_signatures
    import pandas as pd
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    import seaborn as sns

    try:
        os.makedirs(os.path.join(ROOT_DIR, 'results', 'HRD'))
    except OSError:
        pass

    # Loading data
    test_mutations, signatures = get_data('nature2019-panel')
    test_labels, _ = get_data('nature2019-labels')

    # fixing to 0, 1. Removing intermediate hrd
    test_labels = test_labels[:, 0]
    test_data_samples = test_labels != 0
    test_labels = test_labels[test_data_samples]
    test_labels[test_labels == -1] += 1
    models = ['MIX Sig3', 'SigMA', 'TMB', 'NNLS Sig3']
    scores = []
    for model in models:
        if 'MIX Sig3' in model:
            # MIX
            directory = os.path.join(ROOT_DIR, 'experiments/trained_models/MSK-ALL/refit')
            mix = get_best_model(directory, return_model=True)
            test_data = mix.weighted_exposures(test_mutations)

            if 'normalized' not in model:
                test_data *= test_mutations.sum(1, keepdims=True)  # un-normalizing exposures
            test_data = test_data[:, [2]]

        elif model == 'SigMA':
            sigma_output = os.path.join(ROOT_DIR, 'data/nature2019/SigMA_output.tsv')
            all_df = pd.read_csv(sigma_output, sep='\t')
            # In case this is comma separated
            if len(all_df.columns) == 1:
                all_df = pd.read_csv(sigma_output, sep=',')
            test_data = all_df[['exp_sig3']].values
            # test_data = all_df[['Signature_3_mva']].values

        elif model == 'TMB':
            test_data = test_mutations.sum(1, keepdims=True)

        else:
            # NNLS
            test_data = stack_nnls(test_mutations, get_cosmic_signatures()[signatures])
            test_data = test_data[:, [2]]

        test_data = test_data[test_data_samples]

        # Test estimator on data
        prediction_probabilities = test_data
        auc_roc = roc_auc_score(test_labels, prediction_probabilities)
        print(model, 'auc: {:.2f}'.format(auc_roc))
        scores.append([model, '{:.2f}'.format(auc_roc)])
        fpr, tpr, thresholds = roc_curve(test_labels, prediction_probabilities)
        sns.lineplot(fpr, tpr, ci=None)

    plt.legend(models, loc='lower right')
    plt.xlabel('FN')
    plt.ylabel('TP')
    plt.savefig(os.path.join(ROOT_DIR, 'results', 'HRD', 'ROC_HRD.pdf'))
    # plt.show()
    np.savetxt(os.path.join(ROOT_DIR, 'results', 'HRD', 'ROC_AUC.tsv'), scores, '%s', '\t')


@simple_cli.command('ROC_immunotherapy', short_help='Train and save model')
def ROC_immunotherapy():
    from src.analyze_results import get_best_model, stack_nnls
    import matplotlib.pyplot as plt
    from src.utils import get_data, get_cosmic_signatures
    from sklearn.metrics import roc_curve
    from sklearn.metrics import roc_auc_score
    import seaborn as sns

    try:
        os.makedirs(os.path.join(ROOT_DIR, 'results', 'immunotherapy'))
    except OSError:
        pass

    # Loading data
    test_mutations, sigs = get_data('msk2018-LUAD')
    _, sigs = get_data('MSK-ALL')
    test_labels, _ = get_data('msk2018-LUAD-labels')

    models = ['MIX refit Sig4', 'MIX denovo M5', 'TMB', 'NNLS Sig4']
    scores = []
    for model in models:
        if 'MIX refit Sig4' in model:
            # MIX
            directory = os.path.join(ROOT_DIR, 'experiments/trained_models/MSK-ALL/refit')
            mix = get_best_model(directory, return_model=True)
            test_data = mix.weighted_exposures(test_mutations)

            if 'normalized' not in model:
                test_data *= test_mutations.sum(1, keepdims=True)  # un-normalizing exposures
            test_data = test_data[:, [3]]

        elif 'MIX denovo M5' in model:
            # MIX
            directory = os.path.join(ROOT_DIR, 'experiments/trained_models/MSK-ALL/denovo')
            mix = get_best_model(directory, return_model=True)

            test_data = mix.weighted_exposures(test_mutations)

            if 'normalized' not in model:
                test_data *= test_mutations.sum(1, keepdims=True)  # un-normalizing exposures
            test_data = test_data[:, [4]]

        elif model == 'TMB':
            test_data = test_mutations.sum(1, keepdims=True)

        else:
            # NNLS
            test_data = stack_nnls(test_mutations, get_cosmic_signatures()[sigs])
            test_data = test_data[:, [3]]

        # Test estimator on data
        prediction_probabilities = test_data
        auc_roc = roc_auc_score(test_labels, prediction_probabilities)
        print(model, 'auc: {:.2f}'.format(auc_roc))
        scores.append([model, '{:.2f}'.format(auc_roc)])
        fpr, tpr, thresholds = roc_curve(test_labels, prediction_probabilities)
        sns.lineplot(fpr, tpr, ci=None)

    plt.legend(models, loc='lower right')
    plt.xlabel('FN')
    plt.ylabel('TP')
    plt.savefig(os.path.join(ROOT_DIR, 'results', 'immunotherapy', 'ROC_immunotherapy.pdf'))
    # plt.show()
    np.savetxt(os.path.join(ROOT_DIR, 'results', 'immunotherapy', 'ROC_AUC.tsv'), scores, '%s', '\t')


@simple_cli.command('clusters_quality', short_help='Train and save model')
def clusters_quality():

    try:
        os.makedirs(os.path.join(ROOT_DIR, 'results', 'clusters_quality'))
    except OSError:
        pass

    from src.analyze_results import compare_panel_clusters

    compare_panel_clusters()


if __name__ == "__main__":
    simple_cli()
