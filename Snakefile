import os
from src.constants import ROOT_DIR


# Parameters
fast = True # True - run the best seed for each model. False - run all 10 seeds for each model.
max_iterations = 1000

random_seeds = [140296, 142857, 314179, 847662, 3091985, 28021991, 554433, 123456, 654321, 207022]
ds_means = [3, 6, 9, 12, 15, 18]
TRAINED_MODELS = os.path.join(ROOT_DIR, 'experiments', 'trained_models')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
DATA_DIR = os.path.join(ROOT_DIR, 'data')

synthetic_data = expand(os.path.join(DATA_DIR, 'simulated-data/{num_clusters}_{num_sigs}_{num_samples}_{seeds}/mutations.npy'),
        num_clusters=[5, 6, 7, 8, 9], num_sigs=[4], num_samples=[5000], seeds=[140296])

mix_num_clusters = range(1, 21)
mix_num_sigs = range(1, 12)
mix_num_clusters = [str(a).zfill(3) for a in mix_num_clusters]
mix_num_sigs = [str(a).zfill(3) for a in mix_num_sigs]

mix_denovo_msk_models = expand(os.path.join(TRAINED_MODELS, 'MSK-ALL/denovo/mix_{mix_cluster}clusters_{mix_sig}signatures/{seed}seed.json'),
                            mix_cluster=mix_num_clusters, mix_sig=mix_num_clusters, seed=random_seeds)

mix_refit_msk_models = expand(os.path.join(TRAINED_MODELS, 'MSK-ALL/refit/mix_{mix_cluster}clusters_017signatures/{seed}seed.json'),
                            mix_cluster=mix_num_clusters, seed=random_seeds)

mix_num_clusters = range(4, 10)
mix_num_sigs = range(3, 6)
mix_num_clusters = [str(a).zfill(3) for a in mix_num_clusters]
mix_num_sigs = [str(a).zfill(3) for a in mix_num_sigs]

mix_synthetic_models = expand(os.path.join(TRAINED_MODELS, 'simulated_{a}_{b}_5000_140296/denovo/mix_{mix_cluster}clusters_{mix_sig}signatures/{seed}seed.json'),
                            a=[5, 6, 7, 8, 9], b=[4], mix_cluster=mix_num_clusters, mix_sig=mix_num_sigs, seed=random_seeds)

mix_num_clusters = range(1, 5)
mix_num_clusters = [str(a).zfill(3) for a in mix_num_clusters]

mix_BRCA_downsampled_models = expand(os.path.join(TRAINED_MODELS, 'BRCA-ds{a}-part{part}/refit/mix_{mix_cluster}clusters_012signatures/{seed}seed.json'),
                            a=ds_means, part=['1', '2'], mix_cluster=mix_num_clusters, seed=random_seeds)
mix_BRCA_panel_models = expand(os.path.join(TRAINED_MODELS, 'BRCA-panel-part{part}/refit/mix_{mix_cluster}clusters_012signatures/{seed}seed.json'),
                            part=['1', '2'], mix_cluster=mix_num_clusters, seed=random_seeds)
mix_BRCA_full_panel_models = expand(os.path.join(TRAINED_MODELS, 'BRCA-panel/refit/mix_{mix_cluster}clusters_012signatures/{seed}seed.json'),
                            mix_cluster=mix_num_clusters, seed=random_seeds)

mix_OV_downsampled_models = expand(os.path.join(TRAINED_MODELS, 'OV-ds{a}-part{part}/refit/mix_{mix_cluster}clusters_003signatures/{seed}seed.json'),
                            a=ds_means, part=['1', '2'], mix_cluster=mix_num_clusters, seed=random_seeds)
mix_OV_panel_models = expand(os.path.join(TRAINED_MODELS, 'OV-panel-part{part}/refit/mix_{mix_cluster}clusters_003signatures/{seed}seed.json'),
                            part=['1', '2'], mix_cluster=mix_num_clusters, seed=random_seeds)

if fast:
    import data.best_models as best_models

    mix_denovo_msk_models = expand('{models}', models=best_models.BEST_MSK_DENOVO_MODELS)
    mix_refit_msk_models = expand('{models}', models=best_models.BEST_MSK_REFIT_MODELS)

    mix_BRCA_downsampled_models = expand('{models}', models=best_models.BEST_BRCA_DS_MODELS)
    mix_BRCA_panel_models = expand('{models}', models=best_models.BEST_BRCA_PANEL_MODELS)
    mix_BRCA_full_panel_models = expand('{models}', models=best_models.BEST_BRCA_FULL_PANEL_MODELS)

    mix_OV_downsampled_models = expand('{models}', models=best_models.BEST_OV_DS_MODELS)
    mix_OV_panel_models = expand('{models}', models=best_models.BEST_OV_PANEL_MODELS)

    mix_synthetic_models = expand('{models}', models=best_models.BEST_SYNTHETIC_MODELS)


### Results files
results_RE_files = expand(os.path.join(RESULTS_DIR, 'RE', '{dataset}_{error}.tsv'),
                            dataset=['BRCA', 'OV'], error=['mutations', 'exposures'])
results_ami_files = expand(os.path.join(RESULTS_DIR, 'AMI', 'cluster_score_{t}.pdf'), t=['all', 'filtered'])
results_bic_files = expand(os.path.join(RESULTS_DIR, 'BIC', 'MSK-ALL-{t}.pdf'), t=['denovo', 'refit'])
results_hrd_files = os.path.join(RESULTS_DIR, 'HRD', 'ROC_HRD.pdf')
results_sig_similarity_files = expand(os.path.join(RESULTS_DIR, 'signatures_similarity', '{t}-signatures.pdf'), t=range(4, 12))
results_synthetic_files = expand(os.path.join(RESULTS_DIR, 'synthetic', 'simulated_{a}_{b}_{c}_{d}', 'summary.tsv'), a=range(5, 10), b=[4], c=[5000], d=[140296])
results_immunotherapy_files = os.path.join(RESULTS_DIR, 'immunotherapy', 'ROC_immunotherapy.pdf')
results_clusters_quality_files = expand(os.path.join(RESULTS_DIR, 'clusters_quality', 'clusters_quality.pdf'))


rule all:
    input:
        results_RE_files,
        results_ami_files,
        results_bic_files,
        results_sig_similarity_files,
        results_hrd_files,
        results_clusters_quality_files,
        results_immunotherapy_files,
        results_synthetic_files

rule all_no_synthetic:
    input:
        results_RE_files,
        results_ami_files,
        results_bic_files,
        results_sig_similarity_files,
        results_hrd_files,
        results_clusters_quality_files,
        results_immunotherapy_files

rule data_prep:
    input:
        synthetic_data


rule results_HRD:
    output:
        results_hrd_files
    input:
        mix_BRCA_full_panel_models
    shell:
        'python analyze_results_main.py ROC_HRD'


rule results_immunotherapy:
    output:
        results_immunotherapy_files
    input:
        mix_denovo_msk_models,
        mix_refit_msk_models
    shell:
        'python analyze_results_main.py ROC_immunotherapy'


rule results_RE:
    output:
        os.path.join(RESULTS_DIR, 'RE', '{dataset}_{error}.tsv')
    params:
        dataset = '{wildcards.dataset}',
        error = '{wildcards.error}'
    input:
        mix_BRCA_downsampled_models,
        mix_BRCA_panel_models,
        mix_OV_downsampled_models,
        mix_OV_panel_models
    shell:
        'python analyze_results_main.py reconstruction_error --dataset {wildcards.dataset} --error {wildcards.error}'


rule results_sig_learning:
    output:
        os.path.join(RESULTS_DIR, 'signatures_similarity', '{num_sigs}-signatures.pdf')
    params:
        num_sigs = '{wildcards.num_sigs}'
    input:
        mix_denovo_msk_models
    shell:
        'python analyze_results_main.py signatures_similarity --num_sigs {wildcards.num_sigs}'


rule results_ami:
    output:
        results_ami_files
    input:
        mix_denovo_msk_models,
        mix_refit_msk_models
    shell:
        'python analyze_results_main.py AMI'


rule results_BIC:
    output:
        results_bic_files
    input:
        mix_denovo_msk_models,
        mix_refit_msk_models
    shell:
        'python analyze_results_main.py BIC'


rule results_synthesized:
    output:
        os.path.join(RESULTS_DIR, 'synthetic', 'simulated_{num_clusters}_{num_signatures}_{num_samples}_{seed}', 'summary.tsv')
    params:
        num_clusters = '{wildcards.num_clusters}',
        num_signatures = '{wildcards.num_signatures}',
        num_samples = '{wildcards.num_samples}',
        seed = '{wildcards.seed}',
    input:
        synthetic_data,
        mix_synthetic_models
    shell:
        'python analyze_results_main.py analyze_synthetic --dataset simulated_{wildcards.num_clusters}_{wildcards.num_signatures}_{wildcards.num_samples}_{wildcards.seed}'


rule clusters_quality:
    output:
        results_clusters_quality_files
    input:
        mix_BRCA_full_panel_models
    shell:
        'python analyze_results_main.py clusters_quality'



# Preparations
rule synthesize_data:
    input:
        os.path.join(DATA_DIR, 'simulated-data/base_model.json')
    output:
        os.path.join(DATA_DIR, 'simulated-data/{num_clusters}_{num_sigs}_{num_samples}_{seed}/mutations.npy')
    params:
        num_clusters = '{wildcards.num_clusters}',
        num_sigs = '{wildcards.num_sigs}',
        num_samples = '{wildcards.num_samples}',
        seed = '{wildcards.seed}'
    shell:
        'python data_simulation.py simulate --num_clusters {wildcards.num_clusters} --num_signatures {wildcards.num_sigs} --num_samples {wildcards.num_samples} --random_seed {wildcards.seed}'


rule create_synthetic_base_model:
    input:
        os.path.join(TRAINED_MODELS, 'MSK-ALL/denovo/mix_010clusters_006signatures/314179seed.json')
    output:
        os.path.join(DATA_DIR, 'simulated-data/base_model.json')
    shell:
        'python data_simulation.py create_base_model'


# Training
rule denovo_train:
    output:
        os.path.join(TRAINED_MODELS, '{dataset}/denovo/mix_{mix_num_clusters}clusters_{mix_num_signatures}signatures/{seed}seed.json')
    params:
        dataset = '{wildcards.dataset}',
        mix_num_clusters = '{wildcards.mix_num_cluster}',
        mix_num_signatures = '{wildcards.mix_num_signature}',
        seeds = '{wildcards.seed}'
    shell:
        'python main.py train_model --dataset {wildcards.dataset} --num_clusters {wildcards.mix_num_clusters} --use_cosmic 0 --num_signatures {wildcards.mix_num_signatures} --random_seed {wildcards.seed} --max_iterations {max_iterations}'


rule refit_train:
    output:
        os.path.join(TRAINED_MODELS, '{dataset}/refit/mix_{mix_num_clusters}clusters_{mix_num_signatures}signatures/{seed}seed.json')
    params:
        dataset = '{wildcards.dataset}',
        mix_num_clusters = '{wildcards.mix_num_cluster}',
        mix_num_signatures = '{wildcards.mix_num_signature}',
        seeds = '{wildcards.seed}'
    shell:
        'python main.py train_model --dataset {wildcards.dataset} --num_clusters {wildcards.mix_num_clusters} --use_cosmic 1 --num_signatures {wildcards.mix_num_signatures} --random_seed {wildcards.seed} --max_iterations {max_iterations}'