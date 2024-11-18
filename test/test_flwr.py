from flwr.common import Context, ndarrays_to_parameters, EvaluateIns
from src.client_app import client_fn
from src.task import get_parameters, NERLightningModule

def test_client_fn():
    node_config = {"partition-id": 0,
                   "num-partitions": 2}
    run_config = {"max-epochs": 1}
    context = Context(node_id=0, node_config=node_config, state=None, run_config=run_config)
    client = client_fn(context)
    print(client)
    
def test_evaluate():
    node_config = {"partition-id": 0,
                   "num-partitions": 2}
    run_config = {"max-epochs": 1}
    context = Context(node_id=0, node_config=node_config, state=None, run_config=run_config)
    
    client = client_fn(context)
    ndarrays = get_parameters(NERLightningModule())
    global_model_init = ndarrays_to_parameters(ndarrays)
    print(client.evaluate(EvaluateIns(parameters=global_model_init, config=None))) # TODO: switch to a smaller model
    