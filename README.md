# NERF Project: Federated Learning for Named Entity Recognition

This project implements a federated learning setup to train a language model for Named Entity Recognition (NER) in multiple scandinavian languages: Swedish, Norwegian and Danish.

## Table of Contents

1. [Setup](#setup)
2. [Quickstart Example](#quickstart-example)
3. [Complete example](#complete-example)
4. [Other tutorials](#other-tutorials)
    - [Running batch job on Alvis](#running-batch-job-on-alvis)
    - [Running Tests with pytest](#running-tests-with-pytest)

---

## Setup

Before running the [quickstart Example](#quickstart-example) or the [complete example](#complete-example), ensure that the required packages are installed and you are logged into your Wandb account.

1. Load required modules and activate the virtual environment:

```bash
module purge
module load Python/3.10.8-GCCcore-12.2.0
python -m venv /mimer/NOBACKUP/groups/naiss2024-22-1455/venvs/venv_example_lightning
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

1. Download the quickstart repository:

```bash
git clone --depth=1 https://github.com/adap/flower.git _tmp \
        && mv _tmp/examples/quickstart-pytorch-lightning . \
        && rm -rf _tmp && cd quickstart-pytorch-lightning
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


## Complete example

For a more complete federated learning setup, tune the model type, number of training epochs, and other parameters in the `pyproject.toml` file. These examples require GPU access (e.g., via the Alvis cluster).

Make sure the virtual environment is activated and then use: 

```bash
flwr run .
```

### Run a specific file

For all of the following commands, ensure that the virtual environment is activated. You can run custom Python files with specific parameters as needed.
1. Example command to run a specific file:
```bash
python src/file --parameters x
```
2. Example files to run:
    - Baseline run (`src/run_baseline.py`)
    Run the baseline model. You can specify the training datasets (one or more) and the test dataset. Available datasets include:
        - `da_ddt` (danish)
        - `sv_talbanken` (swedish)
        - `nno_norne` (norwegian)
        - `nob_norne` (norwegian)
          
    You can also monitor the validation loss for all datasets during baseline training using the `--monitor_all_val_losses` flag.
   
    Usecase example:
    ```bash
    python src/run_baseline.py --train_datasets da_ddt sv_talbanken --test_dataset da_ddt --monitor_all_val_losses
    ```
    - Federated Learning (`src/federated_learning.py`)
    Run a non-simulated federated learning setup using multiple GPUs on the same machine. You can specify:
        - `num_clients`: Number of clients (default: 2)
        - `model_name`: Model name (default: FacebookAI/xlm-roberta-base)
        - `max_epochs`: Number of epochs per round (default: 1)
        - `num_rounds`: Number of communication rounds (default: 10)
    Usecase example:
    ```bash
    python src/federated_learning.py --num_clients 4 --max_epochs 3
    ```


---

## Other tutorials

### Run batch job on Alvis
Used to get the federated training results:

- For 2 datasets: [scripts/alvis_roberta_base_2_dsets.sh](scripts/alvis_roberta_base_2_dsets.sh).
- For 3 datasets: [scripts/alvis_roberta_base_3_dsets.sh](scripts/alvis_roberta_base_3_dsets.sh).
- For all four datasets: [scripts/alvis_roberta_base_all_dsets.sh](scripts/alvis_roberta_base_all_dsets.sh).

Run them using `sbatch <your-script-here>` on Alvis. Also, make sure to edit the `[tool.flwr.federations]` setting in the `pyproject.toml` file for each run as described in each script.

### Run

Tune the run arguments, e.g. model type and number of training epochs, using [pyproject.toml](pyproject.toml). 

Make sure the virtual environment is activated and then use: 

```bash
flwr run .
```

### Running Tests with pytest

To run tests for the project, use the following command:

```bash
python -m pytest test
```
