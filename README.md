This project was done
# NERF

In this project, we implemented a federated learning setup to train a language model for named entity recognition (NER) on different languages (Swedish, Norwegian, Danish, Icelandic etc.)

## Tutorial: how to run our code

### Setup
Install the required packages (if necessary) and log in to your wandb account, as follows:

```bash
module purge
module load Python/3.10.8-GCCcore-12.2.0
source /mimer/NOBACKUP/groups/naiss2024-22-1455/venvs/venv_example_lightning/bin/activate

pip install -e . # you should only need to run this once
wandb login  
```


### Run flwr

Tune the run arguments, e.g. model type and number of training epochs, using [pyproject.toml](pyproject.toml). 

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

