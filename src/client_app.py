"""pytorchlightning_example: A Flower / PyTorch Lightning app."""

from logging import INFO, DEBUG
import torch
import pytorch_lightning as pl
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.logger import log

from src.task import (
    NERLightningModule,
    get_parameters,
    load_data,
    set_parameters,
)


class FlowerClient(NumPyClient):
    def __init__(self, train_loader, val_loader, test_loader, max_epochs):
        self.model = NERLightningModule()
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.max_epochs = max_epochs
        
        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)  # send model to device
        
        log(INFO, f"Client initialized.")

    def fit(self, parameters, config):
        """Train the model with data of this client."""
        set_parameters(self.model, parameters)
        
        log(INFO, f"Client is doing fit() with config: {config}")
        trainer = pl.Trainer(max_epochs=self.max_epochs, enable_progress_bar=False)
        trainer.fit(self.model.to(self.device), self.train_loader, self.val_loader)

        return get_parameters(self.model), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        set_parameters(self.model, parameters)

        trainer = pl.Trainer(enable_progress_bar=False)
        results = trainer.test(self.model.to(self.device), self.test_loader)
        loss = results[0]["test_loss"]

        return loss, len(self.test_loader.dataset), {}


def client_fn(context: Context) -> Client:
    """Construct a Client that will be run in a ClientApp."""
    # raise ValueError("Can you see this?") # can't
    log(INFO, f"client_fn called with context {context}") # can't see this either

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    train_loader, val_loader, test_loader = load_data(partition_id)

    # Read run_config to fetch hyperparameters relevant to this run
    max_epochs = context.run_config["max-epochs"]
    return FlowerClient(train_loader, val_loader, test_loader, max_epochs).to_client()


app = ClientApp(client_fn=client_fn)
