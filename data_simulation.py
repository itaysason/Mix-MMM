from src.utils import save_json
import click
from src.utils import get_model, load_json
from src.constants import ROOT_DIR
import numpy as np
from src.models.Mix import Mix
import os


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(chain=False, context_settings=CONTEXT_SETTINGS)
@click.option('--debug/--no-debug', default=False, help="Default is --no-debug.")
@click.option('-v', '--verbosity', default=0, help='Verbosity level (0-3).')
def simple_cli(debug, verbosity):
    return


@simple_cli.command('simulate', short_help='Simulate data using MIX')
@click.option('--num_clusters', type=int)
@click.option('--num_signatures', type=int)
@click.option('--avg_num_mut', type=int)
@click.option('--num_samples', type=int)
def simulate(num_clusters, num_signatures, avg_num_mut, num_samples):
    np.random.seed(3141592)
    base_model = get_model(load_json(os.path.join(ROOT_DIR, 'data', 'simulated-data', 'base_model.json'))['parameters'])
    if num_clusters > base_model.num_clusters:
        raise ValueError('num_clusters cannot be larger than base_model.num_clusters ({})'.format(base_model.num_clusters))
    if num_signatures > base_model.num_topics:
        raise ValueError('num_clusters cannot be larger than base_model.num_topics ({})'.format(base_model.num_topics))

    clusters = np.random.choice(base_model.num_clusters, size=num_clusters, replace=False, p=base_model.w)
    pi = base_model.pi[clusters]
    w = base_model.w[clusters]
    w /= w.sum()
    prob_sig = np.dot(w, pi)
    signatures = np.random.choice(base_model.num_topics, size=num_signatures, replace=False, p=prob_sig)

    pi = pi[:, signatures]
    pi /= pi.sum(1, keepdims=True)
    e = base_model.e[signatures]
    model = Mix(num_clusters, num_signatures, init_params={'w': w, 'pi': pi, 'e': e})
    sample_sizes = np.random.poisson(avg_num_mut, num_samples)
    while np.min(sample_sizes) == 0:
        bad_samples = np.where(sample_sizes == 0)[0]
        num_bad_samples = len(bad_samples)
        sample_sizes[bad_samples] = np.random.poisson(avg_num_mut, num_bad_samples)
    clusters, signatures, mutations = model.sample(sample_sizes)

    curr_dir = os.path.join(ROOT_DIR, 'data', 'simulated-data', '{}_{}_{}_{}'.format(num_clusters, num_signatures, avg_num_mut, num_samples))
    try:
        os.makedirs(curr_dir)
    except OSError:
        pass

    # Save model, base data
    save_json(os.path.join(curr_dir, 'full_simulated'), {'clusters': clusters, 'signatures': signatures,
                                                         'mutations': mutations})
    parameters = model.get_params()

    parameters['w'] = parameters['w'].tolist()
    parameters['pi'] = parameters['pi'].tolist()
    parameters['e'] = parameters['e'].tolist()

    save_json(os.path.join(curr_dir, 'model'), parameters)

    # Transform the basic data into mutation matrix
    mutation_mat = np.zeros((num_samples, 96), dtype='int')
    for i in range(num_samples):
        a, b = np.unique(mutations[i], return_counts=True)
        mutation_mat[i, a] = b

    np.save(os.path.join(curr_dir, 'mutations'), mutation_mat)


@simple_cli.command('create_base_model', short_help='Create the model used to simulate data')
def create_base_model():
    try:
        os.makedirs(os.path.join(ROOT_DIR, 'data/simulated-data'))
    except OSError:
        pass
    base_model = load_json(os.path.join(ROOT_DIR, 'experiments/trained_models/MSK-ALL/denovo/mix_010clusters_006signatures/314179seed.json'))
    save_json(os.path.join(ROOT_DIR, 'data/simulated-data/base_model'), base_model)


if __name__ == "__main__":
    simple_cli()
