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
    """
    A subclass of the FedAvg strategy that overrides aggregation methods to enforce stricter handling 
    of client failures. If any failures occur during aggregation, the process halts and raises the 
    first encountered failure.

    Methods:
        aggregate_fit: Overrides the default `aggregate_fit` method to prevent aggregation if failures 
                       are detected and to use the superclass implementation otherwise.
        aggregate_evaluate: Overrides the default `aggregate_evaluate` method to prevent aggregation if 
                            failures are detected, while also logging the aggregated loss after successful 
                            evaluation.
    """
    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Union[tuple[ClientProxy, FitRes], BaseException]],
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """
        Aggregates fit results using a weighted average while enforcing failure handling.

        If any failures occur during the aggregation process, the method raises the first failure 
        encountered and skips aggregation.

        Args:
            server_round (int): The current round of federated training.
            results (list): A list of tuples containing client proxies and their respective fit results.
            failures (list): A list of tuples containing client proxies and their failed fit results or 
                             exceptions raised during the process.

        Returns:
            tuple: A tuple containing the aggregated parameters (or `None` if no results are provided) 
                   and a dictionary of aggregation metadata.

        Raises:
            BaseException: The first exception in the `failures` list if any failures are present.
        """
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
        """
        Aggregates evaluation losses using a weighted average while enforcing failure handling.

        If any failures occur during the aggregation process, the method raises the first failure 
        encountered and skips aggregation. Logs the aggregated loss after successful aggregation.

        Args:
            server_round (int): The current round of federated evaluation.
            results (list): A list of tuples containing client proxies and their respective evaluation results.
            failures (list): A list of tuples containing client proxies and their failed evaluation results 
                             or exceptions raised during the process.

        Returns:
            tuple: A tuple containing the aggregated evaluation loss (or `None` if no results are provided) 
                   and a dictionary of aggregation metadata.

        Raises:
            BaseException: The first exception in the `failures` list if any failures are present.
        """
        # Do not aggregate if there are failures and failures are not accepted
        if len(failures) > 0:
            log(INFO, "Received failures. Reraising the first one:")
            raise failures[0]
        if not results:
            return None, {}

        aggregated_metrics= super().aggregate_evaluate(server_round, results, failures)
        aggregated_loss, _ = aggregated_metrics
        log(INFO, f"Round {server_round}, aggregated loss: {aggregated_loss}")
        return aggregated_metrics

def fit_config(server_round: int):
    """
    Generate the training configuration dictionary for a specific server round.

    This function provides configuration parameters to be passed to the federated learning 
    clients during the `fit` phase of each round. It can be used to define round-specific 
    parameters such as learning rate schedules, batch sizes, or other hyperparameters.

    Args:
        server_round (int): The current round of training.

    Returns:
        dict: A dictionary containing the training configuration for the given round.
              Includes:
              - "current_round" (int): The current server round number.
    """
    """Return training configuration dict for each round.

    """
    config = {
        "current_round": server_round,
    }
    return config

def server_fn(context: Context) -> ServerAppComponents:
    """
    Construct and initialize the server application components.

    This function sets up the federated learning server by:
    - Defining the global model's initial state.
    - Configuring the federated learning strategy (e.g., `FedAvgNoFail`) with custom behavior.
    - Specifying the server configuration, including the number of training rounds.

    Args:
        context (Context): The runtime context of the server application, which includes 
                           run-specific configurations and parameters.

    Returns:
        ServerAppComponents: A container holding the configured federated learning strategy 
                             and server configuration.

    Notes:
        - The initial global model parameters are derived from a `NERLightningModule` 
          initialized with the specified model name (defaulting to `distilbert/distilbert-base-multilingual-cased`).
        - The number of server rounds is configurable via `context.run_config["num-server-rounds"]`.
        - The `FedAvgNoFail` strategy enforces strict failure handling and uses a custom 
          fit configuration callback (`fit_config`).
    """
    log(INFO, f"Starting server with run context: {context}")
    
    # load a model_name if one has been defined, otherwise use the default model (DistilBERT-base)
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
        accept_failures=False,
        on_fit_config_fn=fit_config
    )

    # Construct ServerConfig
    num_rounds = 1
    if "num-server-rounds" in context.run_config:
        num_rounds = context.run_config["num-server-rounds"]
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
