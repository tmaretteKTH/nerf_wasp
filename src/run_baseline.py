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

# The following are the datasets used for our project (see https://github.com/UniversalNER for details)
DATA_NAMES = ["da_ddt", "sv_talbanken", "nno_norne", "nob_norne"]

class BaselineNERLightningModule(NERLightningModule):
    """
    A PyTorch Lightning module for Named Entity Recognition (NER) using a pretrained transformer model.

    This module overwrites the validation_step()-method in the NERLightningModule defined in task.py in
    order to monitor validation loss on datasets we are not training and evaluating on.
    """
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        A function for monitoring the validation loss using several dataloaders.

        This function calls the _evaluate()-method of the NERLightningModule, assigning the stage based 
        on the ID of the dataloader.

        Args:
            batch (dict): A batch of input data with keys `input_ids`, `attention_mask`, and `labels`.
            batch_idx (int): Index of the current batch.
            dataloader_idx (int): Index of the current dataloader.
        """
        if dataloader_idx == 0:
            self._evaluate(batch, batch_idx, "val")
        else:
            dataset_name = self.trainer.val_dataloaders[dataloader_idx].dataset.name
            self._evaluate(batch, batch_idx, f"{dataset_name}_val")


class ShuffleDataLoaderCallback(Callback):
    """
    A callback for recreating and shuffling the DataLoader at the beginning of each epoch.

    This callback is used for the larger Norwegian datasets, so that the data gets shuffled
    at the beginning of each epoch to avoid training on the same data in every epoch. This is not 
    needed in the federated setup because the data gets loaded (and thus shuffled) 
    at the beginning of each round.

    Attributes:
        train_loader (DataLoader): The training dataloader to be shuffled.
    """
    def __init__(self, train_loader):
        self.train_loader = train_loader

    def on_train_epoch_start(self, trainer, pl_module):
        """
        A function for shuffling the training data at the beginning of each epoch.

        This function takes the current train_loader and creates a new one by shuffling its dataset. 
        The new train loader then gets assigned to trainer.fit_loop._data_loader_iter.
        """
        new_train_loader = DataLoader(
            dataset=self.train_loader.dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=True,  # Ensure data is shuffled
            num_workers=self.train_loader.num_workers,
        )
        
        trainer.fit_loop._data_loader_iter = iter(new_train_loader)


def run(model_name = "FacebookAI/xlm-roberta-base", train_datasets = DATA_NAMES, test_dataset = "da_ddt",
        max_epochs = 10, monitor_all_val_losses = True) -> None:
    """
    The main function for training the baseline model.

    This function loads the data, sets up callbacks, learning rate monitoring and logging, prepares
    the trainer, runs training and thereafter evaluates the results on the test set.

    Args:
        model_name (str): The name of the model used for training the baseline.
        train_datasets (List[str]): A list of datasets to train on.
        test_dataset (str): The dataset to validate and test on.
        max_epochs (int): The number of epochs to train for.
        monitor_all_val_losses (bool): Whether to monitor the validation loss on datasets we are not training on.
    """
    
    print("Training on the following datasets:", train_datasets)
    print("Testing on the following dataset:", test_dataset)

    #Load training data
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
    
    # Load evaluation data
    _, val_loader, test_loader = load_data(test_dataset, model_name=model_name)

    # If we want to monitor the validation loss on other datasets than the test_dataset,
    # we load the other datasets here
    # This also affects the wandb run name, the early stopping criteria and if we use the
    # NERLightningModule from task.py or the BaselineNERLightningModule defined above
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

    # Set up wandb logging, learning rate monitoring and the callback for shuffling data
    wandb_logger = WandbLogger(project="nerf_wasp", name=wandb_run_name)
    wandb_logger.experiment.config.update({"model": model_name, "train_data": train_datasets,
                                            "test_data": test_dataset})
    lr_monitor = LearningRateMonitor(logging_interval='step')
    shuffle_callback = ShuffleDataLoaderCallback(train_loader)

    # Prepare trainer
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
    
    # Train and evaluate on the validation set
    if monitor_all_val_losses:
        trainer.fit(model=model, 
                    train_dataloaders=train_loader, 
                    val_dataloaders = all_val_loaders)
    else:
        trainer.fit(model=model, 
                    train_dataloaders=train_loader, 
                    val_dataloaders = val_loader)
    
    # Evaluate on the test set
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

    # Run the script
    run(model_name = args.model_name, 
        train_datasets = args.train_datasets, 
        test_dataset = args.test_dataset,
        monitor_all_val_losses = args.monitor_all_val_losses)
