import argparse
import logging
from logging import INFO, DEBUG, ERROR
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from datasets import concatenate_datasets

from task import (
    NERLightningModule,
    load_data
)

logging.basicConfig(level=INFO)

# "sv_pud"
DATA_NAMES = ["da_ddt", "sv_talbanken", "nno_norne", "nob_norne"]

def run(model_name = "FacebookAI/xlm-roberta-base", train_datasets = DATA_NAMES, test_dataset = "da_ddt",
        max_epochs = 1) -> None:
    
    print("Training on the following datasets:", train_datasets)
    print("Testing on the following dataset:", test_dataset)

    #Prepare data
    if len(train_datasets) == 1 and train_datasets[0] == test_dataset:
        train_loader, val_loader, test_loader = load_data(test_dataset, model_name=model_name)
    elif len(train_datasets) == 1:
        train_loader, _, _ = load_data(train_datasets[0], model_name=model_name)
    elif len(train_datasets) > 1:
        first_dataloader = load_data(train_datasets[0], model_name = model_name)[0]
        batch_size = first_dataloader.batch_size
        num_workers = first_dataloader.num_workers
        train_data = concatenate_datasets([load_data(name, model_name = model_name)[0].dataset for name in train_datasets])
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    _, val_loader, test_loader = load_data(test_dataset, model_name=model_name)

    #Prepare model
    model = NERLightningModule(model_name=model_name)

    wandb_logger = WandbLogger(project="nerf_wasp", name=f"baseline_{'_'.join(train_datasets)}_{test_dataset}")
    wandb_logger.experiment.config.update({"model": model_name, "train_data": train_datasets,
                                            "test_data": test_dataset})
    
    trainer = pl.Trainer(max_epochs=max_epochs, 
                                  logger=wandb_logger, 
                                  accelerator="auto", 
                                  enable_checkpointing=False,
                                  enable_progress_bar=True)
        

    trainer.fit(model=model, 
                train_dataloaders=train_loader, 
                val_dataloaders = val_loader)
    
    # Test model
    trainer.test(model=model, dataloaders=test_loader)


if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser(description="Train a model with specified parameters.")
    
    # Add arguments
    parser.add_argument("--train_datasets", type=str, nargs="+", default=["da_ddt"], help="Train datasets, list of strings")
    parser.add_argument("--test_dataset", type=str, default="da_ddt", help="Test dataset")
    parser.add_argument("--model_name", type=str, default="FacebookAI/xlm-roberta-base", help="Model name")
    parser.add_argument("--max_epochs", type=int, default=1, help="Maximum number of epochs")

    # Parse arguments
    args = parser.parse_args()

    run(model_name = args.model_name, 
        train_datasets = args.train_datasets, 
        test_dataset = args.test_dataset,
        max_epochs = args.max_epochs)
