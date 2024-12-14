# NERF Project: Federated Learning for Named Entity Recognition

This project implements a federated learning setup to train a language model for Named Entity Recognition (NER) in multiple languages, including Swedish, Norwegian, Danish, Icelandic, and others.

## Table of Contents

1. [Setup](#setup)
2. [Quickstart Example](#Quickstart Example)
3. [Longer tutorial](#Longer tutorial)
4. [Other tutorials](#Other tutorials)
    - [Running batch jobs on Alvis](#Running batch jobs on Alvis)
    - [Running Tests with pytest](#Running Tests with pytest)

---

## Setup

Before running the code, ensure that the required packages are installed and you are logged into your Wandb account.

1. Load required modules and activate the virtual environment:

```bash
module purge
module load Python/3.10.8-GCCcore-12.2.0
source /mimer/NOBACKUP/groups/naiss2024-22-1455/venvs/venv_example_lightning/bin/activate
```
2. Install dependencies (run only once):
```
pip install -e .
```
3. Login to Wandb:
```
wandb login  
```
 
---

## Quickstart Example

This is the simplest example to get started with federated learning. It is designed to run on a CPU and is useful when you don't have access to a cluster. For more complete setups (which require GPU access), see the following sections.

1. Download the quickstart repo:

```bash
git clone --depth=1 https://github.com/adap/flower.git _tmp \
        && mv _tmp/examples/quickstart-pytorch-lightning . \
        && rm -rf _tmp && cd quickstart-pytorch-lightning'
cd quickstart-pytorch-lightning
```

2. Fix import deadlock issue:

```bash
pip install datasets==2.21.0
```

3. Run the example:

```bash
flwr run .
```

---


## Longer tutorial

For a more complete federated learning setup, tune the model type, number of training epochs, and other parameters in the `pyproject.toml` file. These examples require GPU access (e.g., via the Alvis cluster).

Make sure the virtual environment is activated and then use: 

```bash
flwr run .
```

### Run a specific file



For all the following, make sure the virtual environment is activated.
You can run a custom python file using

```bash
python src/file --parameters x
```
For instance,
- `src/run_baseline`: run the baseline. You can specify `train_datasets` (one or several datasets) and `test_dataset`, among the 4 following datasets: da_ddt, sv_talbanken, nno_norne, nob_norne. It is possible to monitor the validation loss for all other variable datasets during the baseline training using `monitor_all_val_losses`.
- `src/federated_learning.py`: run a non-simulated setup, using several GPUs on the same machine. You can specify  `num_clients`, `model_name`, `max_epochs` and `num_rounds`.


A final command can look something like this:

```bash
python src/run_baseline.py --train_datasets da_ddt --test_dataset da_ddt --monitor_all_val_losses
```

---

## Other tutorial

### Run batch job on Alvis

To run a batch job on the Alvis cluster, use the script provided:

Use [scripts/alvis_roberta_base_all_dsets.sh](scripts/alvis_roberta_base_all_dsets.sh).

### Running Tests with pytest

To run tests for the project, use the following command:

```bash
python -m pytest test
```
