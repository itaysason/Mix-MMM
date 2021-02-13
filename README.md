## MIX-MMM

### Installation

```sh
conda env create -f environment.yml
source activate MIX-MMM
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

Use the snakefile to reproduce all models used in the paper:

```sh
snakemake all -j{num_cores}
```

This will take a long time so there is a rule that produces less results in less time

```sh
snakemake all -j{num_cores}
```
