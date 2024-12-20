import sys
from pathlib import Path

# Add the project root to the Python path so that the src.task imports in client_app.py and server_app.py
# work fine
sys.path.append(str(Path(__file__).resolve().parent.parent))

import os
import multiprocessing
import time
import logging
import argparse
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)

def start_server(num_rounds, model_name, num_clients):
    """
    Set the necessary environment variables for the server and thereafter run the server app script.

    Args:
        num_rounds (int): The total number of federated learning rounds to run.
        model_name (str): The name of the model used for federated learning.
        num_clients (int): The total number of clients to use for federated learning
    
    Behavior:
        - Sets the environment variables necessary for starting the server.
        - Runs the server script.
    """
    os.environ["NUM_ROUNDS"] = str(num_rounds)
    os.environ["MODEL_NAME"] = model_name
    os.environ["NUM_CLIENTS"] = str(num_clients)
    os.system("python src/server_app_federated_nonsimulated.py")

def start_client(client_id, gpu_id, model_name, max_epochs):
    """
    Set the necessary environment variables for the current client and thereafter run the client app script.

    Args:
        client_id (int): The client and partition ID of the current client.
        gpu_id (int): The GPU ID (CUDA_VISIBLE_DEVICES) for the current client.
        model_name (str): The name of the model used for federated learning.
        max_epochs (int): The number of epochs to run in each round.
    
    Behavior:
        - Sets the environment variables necessary for starting the current client.
        - Runs the client script.
    """
    os.environ["CLIENT_ID"] = str(client_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Consistent GPU variable
    os.environ["MODEL_NAME"] = model_name
    os.environ["MAX_EPOCHS"] = str(max_epochs)
    os.system("python src/client_app_federated_nonsimulated.py")


if __name__ == "__main__":

    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(description="Federated Learning Orchestrator")
    parser.add_argument("--num_clients", type=int, default=2, help="Number of clients")
    parser.add_argument("--model_name", type=str, default="FacebookAI/xlm-roberta-base", help="Model name")
    parser.add_argument("--max_epochs", type=int, default=1, help="Number of epochs per round")
    parser.add_argument("--num_rounds", type=int, default=10, help="Number of communication rounds")
    args = parser.parse_args()

    # Use the parsed arguments
    num_clients = args.num_clients
    model_name = args.model_name
    max_epochs = args.max_epochs
    num_rounds = args.num_rounds

    # Check if there is at least 1 GPU per client
    available_gpus = torch.cuda.device_count()
    if num_clients > available_gpus:
        logging.warning(f"Number of clients ({num_clients}) exceeds available GPUs ({available_gpus}).")

    # Start server
    logging.info(f"Starting server with {num_rounds} rounds using model {model_name}...")
    server_process = multiprocessing.Process(target=start_server, args=(num_rounds, model_name, num_clients), daemon=True)
    server_process.start()

    # This is needed so that the clients don't try to connect to the server before the
    # server has been properly started.
    time.sleep(10)

    
    # Here we start all of the clients
    logging.info(f"Starting {num_clients} clients with model {model_name} for {max_epochs} epochs...")
    client_processes = []
    for client_id in range(num_clients):
        gpu_id = client_id % available_gpus
        process = multiprocessing.Process(
            target=start_client, args=(client_id, gpu_id, model_name, max_epochs), daemon=True
        )
        process.start()
        client_processes.append(process)

    # Wait for all processes to finish
    try:
        server_process.join()
        for process in client_processes:
            process.join()
    except KeyboardInterrupt:
        logging.info("Interrupted by user. Cleaning up...")
        server_process.terminate()
        for process in client_processes:
            process.terminate()