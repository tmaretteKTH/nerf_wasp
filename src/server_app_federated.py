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


class FedAvgNoFail(FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if len(failures) > 0:
            log(INFO, "Received failures. Reraising the first one:")
            raise failures[0]
        if not results:
            return None, {}

        return super().aggregate_fit(server_round, results, failures)
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, EvaluateRes]],
        failures: list[Union[tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if len(failures) > 0:
            log(INFO, "Received failures. Reraising the first one:")
            raise failures[0]
        if not results:
            return None, {}
        
        return super().aggregate_evaluate(server_round, results, failures)

def fit_config(server_round: int):
    """Return training configuration dict for each round."""
    return {
        "current_round": server_round,
    }

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    num_clients = int(os.environ.get("NUM_CLIENTS", 2))

    # Define the model and initialize parameters
    model_name = os.environ.get("MODEL_NAME", "FacebookAI/xlm-roberta-base")
    model = NERLightningModule(model_name=model_name)
    global_model_init = ndarrays_to_parameters(get_parameters(model))

    # Define strategy
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

    # Fetch num_rounds from environment variable
    num_rounds = int(os.environ.get("NUM_ROUNDS", 10))
    config = ServerConfig(num_rounds=num_rounds)

    # Start the server
    start_server(
        server_address="127.0.0.1:8080",
        config=config,
        strategy=strategy,
        grpc_max_message_length=1610612736,
    )