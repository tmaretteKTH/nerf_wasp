import sys
from pathlib import Path

# Add the project root to the Python path so that the src.task imports work fine
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import logging
from logging import INFO, DEBUG, ERROR
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.logger import log
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from flwr.client import start_client

from client_app import FlowerClient
from task import (
    NERLightningModule,
    get_parameters,
    load_data,
    set_parameters,
    BATCH_SIZE,
    NUM_STEPS_PER_ROUND
)

# The following are the datasets used for our project (see https://github.com/UniversalNER for details)
DATA_NAMES = ["da_ddt", "sv_talbanken", "nno_norne", "nob_norne"]

def client_fn(context: Context) -> NumPyClient:
    """
    Construct a Flower federated learning client.

    This function initializes a federated learning client using the partitioned dataset
    and model configuration specified as environment variables (it should be able to use a config in
    the same manner as the simulated setup but we were unable to figure out where that config needs 
    to be specified). It creates a `FlowerClient` instance configured with local training, validation, 
    and test data loaders.

    Args:
        context (Context): Not currently used but flower requires it to be there for the script to run.

    Returns:
        Client: An initialized Flower client ready for federated training and evaluation.
    
    Behavior:
        - Reads the `partition-id` from the environment variables to select the appropriate dataset partition.
        - Configures the pretrained model name, dataset name, and hyperparameters like `max_epochs`.
        - Initializes a `FlowerClient` with the specified model, and dataset.

    """
    partition_id = int(os.environ.get("CLIENT_ID", 0))
    model_name = os.environ.get("MODEL_NAME", "FacebookAI/xlm-roberta-base")
    max_epochs = int(os.environ.get("MAX_EPOCHS", 1))

    dataset_name = DATA_NAMES[partition_id]
    train_loader, val_loader, test_loader = load_data(
        dataset_name, model_name=model_name, batch_size=BATCH_SIZE, num_workers=0,
    )

    return FlowerClient(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        max_epochs=max_epochs,
        model_name=model_name,
        dataset_name=dataset_name,
    )

if __name__ == "__main__":
    # Start the Flower client
    # We had to increase the max message length to be able to use the same model as in the simulation
    start_client(
        server_address="127.0.0.1:8080",
        client_fn=client_fn, 
        grpc_max_message_length=1610612736,
    )