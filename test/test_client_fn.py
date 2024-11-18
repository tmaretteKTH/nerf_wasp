from flwr.common import Context
from src.client_app import client_fn

def test_client_fn():
    node_config = {"partition-id": 0,
                   "num-partitions": 2}
    context = Context(node_id=0, node_config=node_config, state=None, run_config=None)
    client = client_fn(context)
    print(client)