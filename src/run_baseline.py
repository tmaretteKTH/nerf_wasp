import argparse
import logging
from logging import INFO, DEBUG, ERROR
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from datasets import concatenate_datasets
from pytorch_lightning.callbacks import Callback, LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from task import (
    NERLightningModule,
    load_data,
    NUM_STEPS_PER_ROUND
)

logging.basicConfig(level=INFO)

# "sv_pud"
DATA_NAMES = ["da_ddt", "sv_talbanken", "nno_norne", "nob_norne"]

class BaselineNERLightningModule(NERLightningModule):
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            self._evaluate(batch, batch_idx, "val")
        else:
            dataset_name = self.trainer.val_dataloaders[dataloader_idx].dataset.name
            self._evaluate(batch, batch_idx, f"{dataset_name}_val")

# Callback for recreating and shuffling the DataLoader at the beginning of each epoch
class ShuffleDataLoaderCallback(Callback):
    def __init__(self, train_loader):
        self.train_loader = train_loader

    def on_train_epoch_start(self, trainer, pl_module):
        # Recreate DataLoader with shuffled dataset
        new_train_loader = DataLoader(
            dataset=self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=True,  # Ensure data is shuffled
            num_workers=self.train_loader.num_workers,
        )
        
        trainer.fit_loop._data_loader_iter = iter(new_train_loader)


def run(model_name = "FacebookAI/xlm-roberta-base", train_datasets = DATA_NAMES, test_dataset = "da_ddt",
        max_epochs = 10, monitor_all_val_losses = True) -> None:
    
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

    if monitor_all_val_losses:
        # Additional validation datasets
        additional_val_datasets = [dataset for dataset in DATA_NAMES if dataset != test_dataset]
        additional_val_loaders = [load_data(name, model_name=model_name)[1] for name in additional_val_datasets]
        all_val_loaders = [val_loader] + additional_val_loaders
        wandb_run_name = f"baseline_{'_'.join(train_datasets)}_{test_dataset}_monitor_all_val_losses"
        early_stopping_criteria = "val_loss/dataloader_idx_0"
        model = BaselineNERLightningModule(model_name=model_name, mode="baseline")
    else:
        wandb_run_name = f"baseline_{'_'.join(train_datasets)}_{test_dataset}"
        early_stopping_criteria = "val_loss"
        model = NERLightningModule(model_name=model_name, mode="baseline")

    wandb_logger = WandbLogger(project="nerf_wasp", name=wandb_run_name)
    wandb_logger.experiment.config.update({"model": model_name, "train_data": train_datasets,
                                            "test_data": test_dataset})
    lr_monitor = LearningRateMonitor(logging_interval='step')
    shuffle_callback = ShuffleDataLoaderCallback(train_loader)


    trainer = pl.Trainer(max_epochs=max_epochs,
                        logger=wandb_logger,
                        accelerator="auto",
                        callbacks=[
                                    lr_monitor, 
                                    EarlyStopping(monitor=early_stopping_criteria, mode="min", patience=3),
                                    shuffle_callback
                                    ],
                        log_every_n_steps=int(0.1 * NUM_STEPS_PER_ROUND),
                        val_check_interval=int(0.2 * NUM_STEPS_PER_ROUND),
                        enable_checkpointing=False,
                        enable_progress_bar=True,
                        limit_train_batches=NUM_STEPS_PER_ROUND
                        )
    
    if monitor_all_val_losses:
        trainer.fit(model=model, 
                    train_dataloaders=train_loader, 
                    val_dataloaders = all_val_loaders)
    else:
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
    parser.add_argument("--monitor_all_val_losses", action="store_true", default=True, help="Whether you wish to monitor the val loss for all 4 datasets")

    # Parse arguments
    args = parser.parse_args()

    run(model_name = args.model_name, 
        train_datasets = args.train_datasets, 
        test_dataset = args.test_dataset,
        monitor_all_val_losses = args.monitor_all_val_losses)
