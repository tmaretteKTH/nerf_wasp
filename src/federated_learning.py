import os
import multiprocessing
import time
import logging
import argparse
import torch

logging.basicConfig(level=logging.INFO)

def start_server(num_rounds, model_name):
    os.environ["NUM_ROUNDS"] = str(num_rounds)
    os.environ["MODEL_NAME"] = model_name
    os.system("python src/server_app_federated.py")

def start_client(client_id, gpu_id, model_name, max_epochs):
    os.environ["CLIENT_ID"] = str(client_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # Consistent GPU variable
    os.environ["MODEL_NAME"] = model_name
    os.environ["MAX_EPOCHS"] = str(max_epochs)
    os.system("python src/client_app_federated.py")

if __name__ == "__main__":

    # Define and parse command-line arguments
    parser = argparse.ArgumentParser(description="Federated Learning Orchestrator")
    parser.add_argument("--num_clients", type=int, default=2, help="Number of clients")
    parser.add_argument("--model_name", type=str, default="FacebookAI/xlm-roberta-base", help="Model name")
    parser.add_argument("--max_epochs", type=int, default=1, help="Number of epochs per round")
    parser.add_argument("--num_rounds", type=int, default=10, help="Number of communication rounds")
    args = parser.parse_args()

    # Use the parsed arguments with default values
    num_clients = args.num_clients
    model_name = args.model_name
    max_epochs = args.max_epochs
    num_rounds = args.num_rounds

    available_gpus = torch.cuda.device_count()
    if num_clients > available_gpus:
        logging.warning(f"Number of clients ({num_clients}) exceeds available GPUs ({available_gpus}).")

    # Start server
    logging.info(f"Starting server with {num_rounds} rounds using model {model_name}...")
    server_process = multiprocessing.Process(target=start_server, args=(num_rounds, model_name), daemon=True)
    server_process.start()

    time.sleep(10)

    
    # Start clients
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