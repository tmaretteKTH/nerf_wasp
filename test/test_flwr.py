from flwr.common import Context, ndarrays_to_parameters, EvaluateIns, FitIns
import wandb
import torch
from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.server import ServerApp
from src.client_app import client_fn
from src.server_app import server_fn
from src.task import get_parameters, NERLightningModule


def test_client_fn():
    node_config = {"partition-id": 0,
                   "num-partitions": 2}
    run_config = {"max-epochs": 1,
                  "model-name": "FacebookAI/xlm-roberta-large"}
    context = Context(node_id=0, node_config=node_config, state=None, run_config=run_config)
    client = client_fn(context)
    print(client)
    wandb.finish()
    
def test_evaluate():
    # takes about 4 minutes
    node_config = {"partition-id": 0,
                   "num-partitions": 2}
    # switch to a smaller model for the test
    model_name = "distilbert/distilbert-base-multilingual-cased"
    run_config = {"max-epochs": 1,
                  "model-name": model_name}
    context = Context(node_id=0, node_config=node_config, state=None, run_config=run_config)
    
    client = client_fn(context)
    ndarrays = get_parameters(NERLightningModule(model_name=model_name))
    global_model_init = ndarrays_to_parameters(ndarrays)
    print(client.evaluate(EvaluateIns(parameters=global_model_init, config=None)))
    wandb.finish()
    
def test_fit():
    # takes about X minutes
    node_config = {"partition-id": 0,
                   "num-partitions": 2}
    # switch to a smaller model for the test
    model_name = "distilbert/distilbert-base-multilingual-cased"
    run_config = {"max-epochs": 1,
                  "model-name": model_name}
    context = Context(node_id=0, node_config=node_config, state=None, run_config=run_config)
    
    client = client_fn(context)
    ndarrays = get_parameters(NERLightningModule(model_name=model_name))
    global_model_init = ndarrays_to_parameters(ndarrays)
    print(client.fit(FitIns(parameters=global_model_init, config=None)))
    wandb.finish()

def test_run():
    DEVICE = torch.device("cpu")  # Try "cuda" to train on GPU
    print(f"Training on {DEVICE}")

    # Specify the resources each of your clients need
    # If set to none, by default, each client will be allocated 2x CPU and 0x GPUs
    backend_config = {"client_resources": None}
    if DEVICE.type == "cuda":
        backend_config = {"client_resources": {"num_gpus": 1}}

    NUM_PARTITIONS = 2
    # Create the ClientApp
    client = ClientApp(client_fn=client_fn)
    # Create the erverApp
    server = ServerApp(server_fn=server_fn)

    # Run simulation
    run_simulation(
        server_app=server,
        client_app=client,
        num_supernodes=NUM_PARTITIONS,
        backend_config=backend_config,
    ) # TODO: check if we get errors only when using RoBERTa??
    
if __name__ == '__main__':
    # test_evaluate()
    # test_fit()
    test_run()