from src.utils import save_json, get_cosmic_signatures, get_data
from src.experiments import split_train_test_sample_cv, train_mix, train_test_mix
import click
import time
import os


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.group(chain=False, context_settings=CONTEXT_SETTINGS)
@click.option('--debug/--no-debug', default=False, help="Default is --no-debug.")
@click.option('-v', '--verbosity', default=0, help='Verbosity level (0-3).')
def simple_cli(debug, verbosity):
    return


@simple_cli.command('train_model', short_help='Train and save model')
@click.option('--dataset', type=str)
@click.option('--num_clusters', type=int)
@click.option('--use_cosmic', type=int)
@click.option('--num_signatures', type=int, default=0)
@click.option('--random_seed', type=int, default=0)
@click.option('--max_iterations', type=int, default=10000)
@click.option('--epsilon', type=float, default=1e-10)
@click.option('--out_dir', type=str, default='experiments/trained_models')
def train_model(dataset, num_clusters, use_cosmic, num_signatures, random_seed, max_iterations, epsilon, out_dir):
    use_cosmic_dir = 'refit' if use_cosmic else 'denovo'
    dataset_name = dataset
    data, active_signatures = get_data(dataset)
    if use_cosmic:
        num_signatures = len(active_signatures)
        signatures = get_cosmic_signatures()[active_signatures]
    elif num_signatures == 0:
        print('use_cosmic is False and num_signatures is 0, using number of active cosmic signatures {}'.format(len(active_signatures)))
        num_signatures = len(active_signatures)
        signatures = None
    else:
        signatures = None

    model_name = 'mix_' + str(num_clusters).zfill(3) + '_' + str(num_signatures).zfill(3)
    out_dir = os.path.join(out_dir, dataset_name, use_cosmic_dir, model_name)

    try:
        os.makedirs(out_dir)
    except OSError:
        pass

    random_seed = int(time.time()) if random_seed == 0 else random_seed
    out_file = out_dir + "/" + str(random_seed)
    if os.path.isfile(out_file + '.json'):
        print('Experiment with parameters {} {} {} {} {} already exist'.format(
            dataset_name, model_name, use_cosmic, num_signatures, random_seed))
        return

    model, ll = train_mix(data, num_clusters, num_signatures, signatures, random_seed, epsilon, max_iterations)
    parameters = model.get_params()

    parameters['w'] = parameters['w'].tolist()
    parameters['pi'] = parameters['pi'].tolist()
    parameters['e'] = parameters['e'].tolist()

    out = {'log-likelihood': ll, 'parameters': parameters}
    save_json(out_file, out)


@simple_cli.command('sampleCV', short_help='Cross validate over samples')
@click.option('--dataset', type=str)
@click.option('--num_clusters', type=int)
@click.option('--use_cosmic', type=int)
@click.option('--num_folds', type=int)
@click.option('--fold', type=int, default=-1)
@click.option('--num_signatures', type=int, default=0)
@click.option('--shuffle_seed', type=int, default=0)
@click.option('--random_seed', type=int, default=0)
@click.option('--max_iterations', type=int, default=10000)
@click.option('--epsilon', type=float, default=1e-10)
@click.option('--out_dir', type=str, default='experiments/sampleCV')
def sample_cv(dataset, num_clusters, use_cosmic, num_folds, fold, num_signatures, shuffle_seed, random_seed, max_iterations, epsilon, out_dir):

    if fold >= num_folds:
        raise ValueError('num_folds is {} but fold is {}'.format(num_folds, fold))

    dataset_name = dataset
    data, active_signatures = get_data(dataset)
    if use_cosmic:
        num_signatures = len(active_signatures)
        signatures = get_cosmic_signatures()[active_signatures]
    elif num_signatures == 0:
        print('use_cosmic is False and num_signatures is 0, using number of active cosmic signatures {}'.format(
            len(active_signatures)))
        num_signatures = len(active_signatures)
        signatures = None
    else:
        signatures = None

    model_name = 'mix_' + str(num_clusters).zfill(3) + '_' + str(num_signatures).zfill(3)

    for i in range(num_folds):
        if fold >= 0:
            i = fold
        use_cosmic_dir = 'refit' if use_cosmic else 'denovo'
        curr_out_dir = os.path.join(out_dir, dataset_name, use_cosmic_dir, model_name, str(shuffle_seed), str(num_folds), str(i))

        try:
            os.makedirs(curr_out_dir)
        except OSError:
            pass

        random_seed = int(time.time()) if random_seed == 0 else random_seed
        out_file = curr_out_dir + "/" + str(random_seed)
        if os.path.isfile(out_file + '.json'):
            print('Experiment with parameters {} {} {} {} {} {} {} {} already exist'.format(
                dataset_name, model_name, num_folds, use_cosmic, i, num_signatures, shuffle_seed, random_seed))
            continue

        train_data, test_data = split_train_test_sample_cv(data, num_folds, i, shuffle_seed)

        model, train_ll, test_ll = train_test_mix(train_data, test_data, num_clusters, num_signatures, signatures, random_seed, epsilon, max_iterations)
        parameters = model.get_params()

        parameters['w'] = parameters['w'].tolist()
        parameters['pi'] = parameters['pi'].tolist()
        parameters['e'] = parameters['e'].tolist()

        out = {'log-likelihood-train': train_ll, 'log-likelihood-test': test_ll, 'parameters': parameters}
        save_json(out_file, out)
        if fold >= 0:
            break


if __name__ == "__main__":
    simple_cli()
