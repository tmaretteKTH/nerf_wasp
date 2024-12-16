# NERF Project: Federated Learning for Named Entity Recognition

Authors: Jennifer Andersson (Uppsala universitet), Lovisa Hagström (Chalmers tekniska högskola), Kätriin Kukk (Linköpings universitet) and Thibault Marette (Kungliga Tekniska högskolan).


This project implements a federated learning setup to train a language model for Named Entity Recognition (NER) in multiple scandinavian languages: Swedish, Norwegian and Danish.



## Table of Contents
1. [Project deliverable](#project-deliverable)
    - [Project description](#project-description)
    - [Authors contribution](#authors-contribution)
1. [Setup](#setup)
2. [Quickstart Example](#quickstart-example)
3. [Complete example](#complete-example)
4. [Other tutorials](#other-tutorials)
    - [Running batch job on Alvis](#running-batch-job-on-alvis)
    - [Running Tests with pytest](#running-tests-with-pytest)

---

## Project deliverable

The slides used for the oral presentation are available [here](https://docs.google.com/presentation/d/13tEHW9sBmro51u8qW577dTQFy3YLxQSw7boc04vtDoQ/edit?usp=sharing).

### Project description


In this project we employed federated learning to train a language model for named entity recognition (NER) across several scandinavian languages (Swedish, Danish and Norwegian). In this setup, several clients, each with their own dataset, want to collaborate on building a shared NER model. However, due, to privacy regulations (for instance), these clients are unable to exhange their datasets. To address this problem, we leverage federated learning, where each client train their own NER model locally with their data, and only communicate new weights with a global model, shared between all the clients. The global model is then updated by aggregating the received weights, using a federated aggregation strategy.

Our implemenation was deployed on an Alvis node, with each client running on a separate GPU. We evaluated the performance of our federated approahc using precision, recall, and F1-score metrics, and compared it against a baseline model trained without the federated framework.


### Authors contribution

Jennifer helped develop the implementation for federated learning of a RoBERTa model for NER. She also set up the evaluation scheme and was responsible for setting up an outline for the presentation slides and report. She also took on responsibility for the report writing and constructing part of the presentation slides.

Lovisa helped develop the code for training a RoBERTa model on NER datasets in a federated manner. She also contributed to setting up the environment and scripts for running the code on the HPC cluster, Alvis. 

Kätriin helped set up the initial code and the environment for doing federated learning and was responsible for setting up the code for and running the baseline experiments (in run_baseline.py). In addition, she prepared code so that it is possible to run our code in a non-simulated setup (still on the same node).

Thibault helped finding unified high quality datasets for NER tasks, in different languages. He also took responsability of writing a clear and detailed README file for the Github repository, for reproducibility of the results.

All authors participated in the discussion and analysis of the results, and contributed equally to the presentation slides and the written report of the project.

---

## Setup

Before running the [quickstart Example](#quickstart-example) or the [complete example](#complete-example), ensure that the required packages are installed and you are logged into your Wandb account.


1. Load required modules:

```bash
module purge
module load Python/3.10.8-GCCcore-12.2.0
```

2. **One time only** Create the virtual environment:
```bash
python -m venv /mimer/NOBACKUP/groups/naiss2024-22-1455/venvs/venv_example
```

3. Activate the python environment:
```bash
source /mimer/NOBACKUP/groups/naiss2024-22-1455/venvs/venv_example/bin/activate
```

4. **One time only** Install dependencies:
```
pip install -e .
```

5. Login to Wandb:
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
