"""pytorchlightning_example: A Flower / PyTorch Lightning app."""

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
from flwr.common.logger import log
from src.task import NERLightningModule, get_parameters

class FedAvgNoFail(FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        # Do not aggregate if there are failures and failures are not accepted
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
        # Do not aggregate if there are failures and failures are not accepted
        if len(failures) > 0:
            log(INFO, "Received failures. Reraising the first one:")
            raise failures[0]
        if not results:
            return None, {}
        
        return super().aggregate_evaluate(server_round, results, failures)

def server_fn(context: Context) -> ServerAppComponents:
    """Construct components for ServerApp."""
    log(INFO, f"Starting server with run context: {context}")
    
    model_name = "distilbert/distilbert-base-multilingual-cased"
    if "model-name" in context.run_config:
        model_name = context.run_config["model-name"]
    # Convert model parameters to flwr.common.Parameters
    ndarrays = get_parameters(NERLightningModule(model_name))
    global_model_init = ndarrays_to_parameters(ndarrays)

    # Define strategy
    strategy = FedAvgNoFail(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        initial_parameters=global_model_init,
        accept_failures=False
    )

    # Construct ServerConfig
    num_rounds = 1
    if "num-server-rounds" in context.run_config:
        num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
