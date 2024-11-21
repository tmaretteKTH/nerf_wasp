"""pytorchlightning_example: A Flower / PyTorch Lightning app."""

import logging
from logging import INFO, DEBUG, ERROR
import torch
import pytorch_lightning as pl
import flwr
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.logger import log
from pytorch_lightning.loggers import WandbLogger

from src.task import (
    NERLightningModule,
    get_parameters,
    load_data,
    set_parameters,
)

# "sv_pud"
DATA_NAMES = ["da_ddt", "sv_talbanken", "nno_norne", "nob_norne"]

class FlowerClient(NumPyClient):
    def __init__(self, train_loader, val_loader, test_loader, max_epochs, model_name, dataset_name):
        self.model = NERLightningModule(model_name=model_name)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.max_epochs = max_epochs
        self.model_name = model_name
        self.dataset_name = dataset_name
        
        wandb_logger = WandbLogger(project="nerf_wasp", name=f"{self.dataset_name}")
        wandb_logger.experiment.config.update({"model": self.model_name})
        self.trainer = pl.Trainer(max_epochs=self.max_epochs, logger=wandb_logger, accelerator="auto", enable_progress_bar=True)
        
        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device
        

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_parameters(self.model, parameters)
        
        log(INFO, f"Client is doing fit() with config: {config}")
        self.trainer.fit(self.model.to(self.device), self.train_loader, self.val_loader)

        return get_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_parameters(self.model, parameters)

        results = self.trainer.test(self.model.to(self.device), self.test_loader)
        loss = results[0]["test_loss"]

        return loss, len(self.test_loader.dataset), {}


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""
    logger = logging.getLogger("flower")
    logger.info("Test from client")
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    
    model_name = "distilbert/distilbert-base-multilingual-cased"
    if "model-name" in context.run_config:
        model_name = context.run_config["model-name"]
    dataset_name = DATA_NAMES[partition_id]
    train_loader, val_loader, test_loader = load_data(dataset_name, model_name=model_name)

    # Read run_config to fetch hyperparameters relevant to this run
    max_epochs = 1
    if "max-epochs" in context.run_config:
        max_epochs = context.run_config["max-epochs"]
    return FlowerClient(train_loader, val_loader, test_loader, max_epochs, model_name=model_name, dataset_name=dataset_name).to_client()


app = ClientApp(client_fn=client_fn)
