"""pytorchlightning_example: A Flower / PyTorch Lightning app."""

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

from src.task import (
    NERLightningModule,
    get_parameters,
    load_data,
    set_parameters,
    BATCH_SIZE,
    NUM_STEPS_PER_ROUND
)

# our pre-defined data names for the federated learning
# these are mapped to the partition id of the client
DATA_NAMES = ["da_ddt", "sv_talbanken", "nno_norne", "nob_norne"]

class FlowerClient(NumPyClient):
    """
    A client implementation for federated learning using Flower and PyTorch Lightning.

    This class represents a federated learning client, which trains and evaluates a 
    Named Entity Recognition (NER) model using its local dataset. It communicates 
    with a federated server to exchange model parameters and updates during training.

    Attributes:
        model (NERLightningModule): The NER model for token classification.
        train_loader (DataLoader): DataLoader for the training dataset.
        val_loader (DataLoader): DataLoader for the validation dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        max_epochs (int): Maximum number of epochs for local training.
        model_name (str): The name of the pretrained model (e.g., "distilbert/distilbert-base-multilingual-cased").
        dataset_name (str): The name of the dataset assigned to the client.
        trainer (pl.Trainer): PyTorch Lightning trainer for managing the training and evaluation process.
        device (torch.device): The device (CPU or GPU) on which the model is trained and evaluated.

    Methods:
        __init__(train_loader, val_loader, test_loader, max_epochs, model_name, dataset_name):
            Initializes the client with its data, model, and training configurations.

        fit(parameters, config):
            Trains the model using the client's training and validation datasets.
            Args:
                parameters (list): Model parameters received from the server.
                config (dict): Configuration dictionary containing metadata like the current training round.
            Returns:
                tuple: Updated model parameters, the size of the training dataset, and an empty dictionary.

        evaluate(parameters, config):
            Evaluates the model using the client's test dataset.
            Args:
                parameters (list): Model parameters received from the server.
                config (dict): Configuration dictionary (unused in this implementation).
            Returns:
                tuple: The test loss, the size of the test dataset, and an empty dictionary.
    """
    def __init__(self, train_loader, val_loader, test_loader, max_epochs, model_name, dataset_name):
        """
        Initialize the federated client with data loaders, model configurations, and training setup.

        Args:
            train_loader (DataLoader): DataLoader for the training dataset.
            val_loader (DataLoader): DataLoader for the validation dataset.
            test_loader (DataLoader): DataLoader for the test dataset.
            max_epochs (int): Maximum number of training epochs.
            model_name (str): Pretrained model name (e.g., "distilbert/distilbert-base-multilingual-cased").
            dataset_name (str): The name of the dataset assigned to this client.
        """
        self.model = NERLightningModule(model_name=model_name)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.max_epochs = max_epochs
        self.model_name = model_name
        self.dataset_name = dataset_name
        
        wandb_logger = WandbLogger(project="nerf_wasp", name=f"{self.dataset_name}")
        wandb_logger.experiment.config.update({"model": self.model_name})
        lr_monitor = LearningRateMonitor(logging_interval='step')
        self.trainer = pl.Trainer(max_epochs=self.max_epochs, 
                                  max_steps=NUM_STEPS_PER_ROUND,
                                  logger=wandb_logger, 
                                  accelerator="auto", 
                                  callbacks=[lr_monitor, EarlyStopping(monitor="val_loss", mode="min", patience=3)],
                                  log_every_n_steps=int(0.1*NUM_STEPS_PER_ROUND),
                                  val_check_interval=int(0.2*NUM_STEPS_PER_ROUND),
                                  enable_checkpointing=False,
                                  enable_progress_bar=True)
        
        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device
        

    def fit(self, parameters, config):
        """
        Train the model using the client's local dataset.

        This method updates the model's parameters with those received from the server,
        performs local training on the training dataset, and returns the updated 
        parameters along with the size of the training dataset.

        Args:
            parameters (list): Model parameters received from the federated server.
            config (dict): Configuration dictionary containing metadata, including 
                           the current round of training.

        Returns:
            tuple: Updated model parameters, the size of the training dataset, and an empty dictionary.
        """
        # update the train scheduler to the current round
        self.model.current_round = config["current_round"]
            
        set_parameters(self.model, parameters)
        log(DEBUG, f"Client is doing fit() with config: {config}")
        self.trainer.fit(self.model.to(self.device), self.train_loader, self.val_loader)

        return get_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """
        Evaluate the model using the client's local test dataset.

        This method updates the model's parameters with those received from the server
        and computes the test loss on the local test dataset.

        Args:
            parameters (list): Model parameters received from the federated server.
            config (dict): Configuration dictionary (not used in this implementation).

        Returns:
            tuple: The test loss, the size of the test dataset, and an empty dictionary.
        """
        set_parameters(self.model, parameters)

        results = self.trainer.test(self.model.to(self.device), self.test_loader)
        loss = results[0]["test_loss"]

        return loss, len(self.test_loader.dataset), {}


def client_fn(context: Context) -> Client:
    """
    Construct a Flower federated learning client to be run in a `ClientApp`.

    This function initializes a federated learning client using the partitioned dataset
    and model configuration specified in the provided `context`. It creates a 
    `FlowerClient` instance configured with local training, validation, and test data 
    loaders, along with hyperparameters defined in the `run_config`.

    Args:
        context (Context): The run context for the client, which includes node-specific 
                           and run-specific configurations.

    Returns:
        Client: An initialized Flower client ready for federated training and evaluation.
    
    Behavior:
        - Reads the `partition-id` from the `node_config` to select the appropriate dataset partition.
        - Logs the number of GPUs visible and other CUDA settings for debugging.
        - Configures the pretrained model name, dataset name, and hyperparameters like `max_epochs`.
        - Initializes a `FlowerClient` with the specified model, dataset, and hyperparameters.

    Example:
        In a federated setting with Flower, this function is called to create a client 
        instance on each participating node in the distributed system.
    """
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]

    # debugging: ensure that each client has been mapped to a separate GPU
    print(f"Client {partition_id}: Number of GPUs visible: {torch.cuda.device_count()}")
    print(f"Client {partition_id}: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    # load a model_name if one has been defined, otherwise use the default model (DistilBERT-base)
    # (same as in server_app.py)
    model_name = "distilbert/distilbert-base-multilingual-cased"
    if "model-name" in context.run_config:
        model_name = context.run_config["model-name"]
        
    # get the dataset name for the current partition
    dataset_name = DATA_NAMES[partition_id]
    print(f"Client with PID {os.getpid()} is using partition ID {partition_id} and dataset {dataset_name}")
    train_loader, val_loader, test_loader = load_data(dataset_name, model_name=model_name, batch_size=BATCH_SIZE)

    # Read run_config to fetch hyperparameters relevant to this run
    max_epochs = 1
    if "max-epochs" in context.run_config:
        max_epochs = context.run_config["max-epochs"]
    return FlowerClient(train_loader, val_loader, test_loader, max_epochs, model_name=model_name, dataset_name=dataset_name).to_client()


app = ClientApp(client_fn=client_fn)
