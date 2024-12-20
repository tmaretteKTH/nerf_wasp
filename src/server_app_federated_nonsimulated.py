import sys
from pathlib import Path

# Add the project root to the Python path so that the src.task imports work fine
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import logging
from logging import INFO, DEBUG
from typing import Optional, Union
from flwr.common import (
    EvaluateRes,
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server.client_proxy import ClientProxy
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server import start_server
from flwr.common.logger import log
from task import NERLightningModule, get_parameters
from server_app import FedAvgNoFail, fit_config

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Get the number of clients from environment variable
    num_clients = int(os.environ.get("NUM_CLIENTS", 2))

    # Define the model and initialize parameters
    model_name = os.environ.get("MODEL_NAME", "FacebookAI/xlm-roberta-base")
    model = NERLightningModule(model_name=model_name)
    global_model_init = ndarrays_to_parameters(get_parameters(model))

    # Define strategy
    # We set min_available_clients, min_fit_clients and min_evaluate_clients equal to the number of clients
    # specified when starting the script so that all clients are used in all rounds
    strategy = FedAvgNoFail(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        initial_parameters=global_model_init,            
        min_available_clients = num_clients,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        accept_failures=False,
        on_fit_config_fn=fit_config,
    )

    # Fetch num_rounds from environment variable and create ServerConfig
    num_rounds = int(os.environ.get("NUM_ROUNDS", 10))
    config = ServerConfig(num_rounds=num_rounds)

    # Start the server
    # We had to increase the grpc_max_message_length to be able to use the same model as in the simulation
    start_server(
        server_address="127.0.0.1:8080",
        config=config,
        strategy=strategy,
        grpc_max_message_length=1610612736,
    )