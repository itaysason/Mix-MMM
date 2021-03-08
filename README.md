## MIX-MMM

### Installation

```sh
conda env create -f environment.yml
source activate Mix-env
```

### Training the model

```sh
python main.py train_model --dataset _ --num_clusters _ --use_cosmic _ --num_signatures _ --random_seed _ --max_iterations _
```

Available datasets are: ICGC-BRCA, TCGA-OV, MSK-ALL, BRCA-panel, OV-panel and more.
The code can easily be changed to support more datasets as well as getting a file instead of a name of the dataset.

### Use for developers

Developers can load data and precomputed signatures in anyway they want and use as follow

##### Denovo training

```sh
from src.models.Mix import Mix

data # Loaded data in ndarray samplesXmutations
mix = MIX(num_clusters, num_topics)
mix.fit(data) 
```

##### Refit training

```sh
from src.models.Mix import Mix

data # Loaded data in ndarray (samples, mutations)
signatures # Loaded signatures in ndarray (signatures, mutations) 
mix = MIX(num_clusters, num_topics, init_params={e: signatures})
mix.refit(data) 
```

### Reproducing paper results

First to prepare the synthetic data run

```sh
snakemake data_prep -j{num_cores}
```

This will train the Mix used for synthesizing data and synthesize the data. Then use the snakefile to reproduce all models used in the paper:

```sh
snakemake all -j{num_cores}
```

This will take a long time because each model is trained 10 times, so to save time and train only the best seed for
each model, first change the False in line 6 in Snakefile to True (default is True).
This will reduce running time by 10x, but still might take a long time, so if you are willing to drop the synthetic data
results you can run:

```sh
snakemake all_no_synthetic -j{num_cores}
```

To get all results it is actually recommended to run in the following order:

```sh
snakemake all_no_synthetic -j{num_cores}
snakemake data_prep -j{num_cores}
snakemake all -j{num_cores}
```

Here the first command will create most results and in the process will creat the Mix model used to synthesize data,
thus better for parallelization. Another way to produce (most likely) similar results but faster is to
change the default 1000 iterations in line 7 in Snakefile. This will reduce running time but will change the results
(for better or worse). Also note that the best seeds were chosen using 1000 iterations and there is no guarantee they
are the best seeds for other max_iterations.
