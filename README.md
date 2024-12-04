# nerf_wasp

## Run our code 

### Setup
Install the required packages (if necessary) and log in to your wandb account, as follows:

```bash
module purge
module load Python/3.10.8-GCCcore-12.2.0
source /mimer/NOBACKUP/groups/naiss2024-22-1455/venvs/venv_example_lightning/bin/activate

pip install -e . # you should only need to run this once
wandb login  
```

### Run

Tune the run arguments, e.g. model type and number of training epochs, using [pyproject.toml](pyproject.toml). 

Make sure the virtual environment is activated and then use: 

```bash
flwr run .
```

### Run baseline

Make sure the virtual environment is activated and then use for example: 

```bash
python src/run_baseline.py --train_datasets da_ddt --test_dataset da_ddt
```
It is also possible to specify several training datasets like this:

```bash
python src/run_baseline.py --train_datasets da_ddt sv_talbanken --test_dataset da_ddt
```
The datasets available are the following: da_ddt, sv_talbanken, nno_norne, nob_norne

### Run batch job on Alvis

Use [scripts/alvis_roberta_base_all_dsets.sh](scripts/alvis_roberta_base_all_dsets.sh).

## Run tests using pytest

```bash
python -m pytest test
```

## Run quickstart example on Alvis

Follow [this PyTorch-lightning Flower quickstart guide](https://github.com/adap/flower/tree/main/examples/quickstart-pytorch-lightning).

### Set up environment

> Only needs to be done once on Alvis! You should be able to go directly to "Run example".

Before `pip install -e .` in the guide:
```bash
module purge
module load Python/3.10.8-GCCcore-12.2.0
python -m venv /mimer/NOBACKUP/groups/naiss2024-22-1455/venvs/venv_example_lightning
source /mimer/NOBACKUP/groups/naiss2024-22-1455/venvs/venv_example_lightning/bin/activate
```

Small edit to avoid python import deadlock - downgrade datasets from 3.1.0 to 2.21.0.
```bash
cd quickstart-pytorch-lightning
pip install -e .
pip install datasets==2.21.0
```

From the `quickstart-pytorch-lightning` folder:
```bash
flwr run .
```

### Run example

```bash
module purge
module load Python/3.10.8-GCCcore-12.2.0
source /mimer/NOBACKUP/groups/naiss2024-22-1455/venvs/venv_example_lightning/bin/activate

cd quickstart-pytorch-lightning
flwr run .
```

## Datasets

We will use [this dataset](https://huggingface.co/datasets/K2triinK/universal_ner_nordic_FL) for the project.

I found this Universal NER project that came out this year. Some links: [website](https://www.universalner.org/), [research paper](https://arxiv.org/html/2311.09122v2), [git](https://github.com/UniversalNER)

> Universal Named Entity Recognition (UNER) aims to fill a gap in multilingual NLP: high quality NER datasets in many languages with a shared tagset.
> 
> UNER is modeled after the Universal Dependencies project, in that it is intended to be a large community annotation effort with language-universal guidelines. Further, we use the same text corpora as Universal Dependencies.

Basically, it seems to be what [scandiNER](https://huggingface.co/saattrupdan/nbailab-base-ner-scandi) did, but cleaner and more accessible. They used the same annotations guidelines as NorNE (dataset used by scandiNER).


All datas are in [IOB2 format](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging))

They seem to have Swedish/Norwegian/Danish for now. We could replace Icelandic/Faroese by german?


