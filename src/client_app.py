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
        """Train the model with data of this client."""
        # update the train scheduler to the current round
        self.model.current_round = config["current_round"]
            
        set_parameters(self.model, parameters)
        log(DEBUG, f"Client is doing fit() with config: {config}")
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
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]

    print(f"Client {partition_id}: Number of GPUs visible: {torch.cuda.device_count()}")
    print(f"Client {partition_id}: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    
    model_name = "distilbert/distilbert-base-multilingual-cased"
    if "model-name" in context.run_config:
        model_name = context.run_config["model-name"]
    dataset_name = DATA_NAMES[partition_id]
    train_loader, val_loader, test_loader = load_data(dataset_name, model_name=model_name, batch_size=BATCH_SIZE)

    # Read run_config to fetch hyperparameters relevant to this run
    max_epochs = 1
    if "max-epochs" in context.run_config:
        max_epochs = context.run_config["max-epochs"]
    return FlowerClient(train_loader, val_loader, test_loader, max_epochs, model_name=model_name, dataset_name=dataset_name).to_client()


app = ClientApp(client_fn=client_fn)
