"""pytorchlightning_example: A Flower / PyTorch Lightning app."""

import logging
from collections import OrderedDict
from typing import Any

import pytorch_lightning as pl
import torch
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_metric, load_dataset
from datasets.utils.logging import disable_progress_bar
from transformers import get_linear_schedule_with_warmup

MAX_SAMPLES_PER_ROUND = 4300 # tuned to ddt and talbanken
BATCH_SIZE = 32
NUM_STEPS_PER_ROUND = int(MAX_SAMPLES_PER_ROUND/BATCH_SIZE) + 1

disable_progress_bar()

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

class NERLightningModule(pl.LightningModule):
    """
    A PyTorch Lightning module for Named Entity Recognition (NER) using a pretrained transformer model.

    This module leverages a transformer-based model for token classification tasks, such as identifying 
    entities like Person (PER), Organization (ORG), and Location (LOC) in text data. It supports training, 
    validation, and evaluation, with metrics computed using the `seqeval` library.

    Attributes:
        model (AutoModelForTokenClassification): The transformer model for token classification.
        learning_rate (float): The learning rate for the optimizer.
        metric (Metric): The evaluation metric, `seqeval`, for NER tasks.
        label_list (list): The list of BIO style NER labels used for predictions and ground truth.
        current_round (int): Tracks the current training round (useful in federated learning setups).
        mode (str): Specifies the mode of operation, e.g., "federated" or standalone.
    """
    def __init__(self, 
                 model_name="distilbert/distilbert-base-multilingual-cased", 
                 cache_dir="/mimer/NOBACKUP/groups/naiss2024-22-1455/project-data/nerf_wasp/.cache", 
                 num_labels=7, 
                 learning_rate=2e-5,
                 mode = "federated") -> None:
        """
        Initializes the NERLightningModule with a pretrained model and relevant parameters.

        Args:
            model_name (str): Name or path of the pretrained transformer model.
            cache_dir (str): Directory to cache the pretrained model.
            num_labels (int): Number of NER labels (e.g., 7 for BIO format with 3 entity types).
            learning_rate (float): Learning rate for training the model.
            mode (str): Operational mode (e.g., "federated" or "standalone").
        """
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=cache_dir, num_labels=num_labels) 
        self.learning_rate = learning_rate
        self.metric = load_metric("seqeval", trust_remote_code=True)
        self.label_list = ["O", "B-PER", "I-PER", "B-ORG","I-ORG","B-LOC", "I-LOC"]
        self.current_round = 0
        self.mode = mode
        
    def forward(self, input_ids, attention_mask, labels=None) -> Any:
        """
        Defines the forward pass of the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            attention_mask (torch.Tensor): Attention mask for padding tokens.
            labels (torch.Tensor, optional): Ground truth labels for token classification.

        Returns:
            Any: Model outputs, including logits and optionally loss.
        """
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)
    
    def training_step(self, batch, batch_idx):
        """
        Performs a single training step.

        Args:
            batch (dict): A batch of input data with keys `input_ids`, `attention_mask`, and `labels`.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: The loss value for the current batch.
        """
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss    
    
    def _evaluate(self, batch, batch_idx, stage=None):
        """
        Helper method for evaluation during validation or testing.

        Args:
            batch (dict): A batch of input data with keys `input_ids`, `attention_mask`, and `labels`.
            batch_idx (int): Index of the current batch.
            stage (str, optional): The current evaluation stage, e.g., "val" or "test".

        Logs:
            Loss for the batch during the specified stage.
        """
        outputs = self(**batch)
        loss = outputs.loss
        logits = outputs.logits
        # use a greedy decoding to get model predictions
        predictions = torch.argmax(logits, dim=-1)
        labels = batch["labels"]

        #Preprocess predictions and labels before calling seqeval
        # Remove ignored index (-100)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # log evaluation results
        self.metric.add_batch(predictions=true_predictions, references=true_labels)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        """
        Performs a single validation step.

        Args:
            batch (dict): A batch of input data with keys `input_ids`, `attention_mask`, and `labels`.
            batch_idx (int): Index of the current batch.
        """
        self._evaluate(batch, batch_idx, "val")
        # return loss
        
    def validation_epoch_end(self, outputs):
        """
        Computes and logs validation metrics at the end of the validation epoch.

        Args:
            outputs (list): List of outputs from `validation_step`.

        Logs:
            - Overall F1, precision, and recall.
            - F1, precision, and recall for specific entity types (PER, ORG, LOC).
        """
        metrics = self.metric.compute()

        #Overall results
        self.log("val_f1", metrics["overall_f1"], prog_bar=True)
        self.log("val_precision", metrics["overall_precision"], prog_bar=True)
        self.log("val_recall", metrics["overall_recall"], prog_bar=True)
        #Person
        try:
            self.log("PER_val_f1", metrics["PER"]["f1"], prog_bar=True)
            self.log("PER_val_precision", metrics["PER"]["precision"], prog_bar=True)
            self.log("PER_val_recall", metrics["PER"]["recall"], prog_bar=True)
        except KeyError:
            pass
        #Organization
        try:
            self.log("ORG_val_f1", metrics["ORG"]["f1"], prog_bar=True)
            self.log("ORG_val_precision", metrics["ORG"]["precision"], prog_bar=True)
            self.log("ORG_val_recall", metrics["ORG"]["recall"], prog_bar=True)
        except KeyError:
            pass
        #Location
        try:
            self.log("LOC_val_f1", metrics["LOC"]["f1"], prog_bar=True)
            self.log("LOC_val_precision", metrics["LOC"]["precision"], prog_bar=True)
            self.log("LOC_val_recall", metrics["LOC"]["recall"], prog_bar=True)
        except KeyError:
            pass
        

    def test_step(self, batch, batch_idx):
        """
        Performs a single test step.

        Args:
            batch (dict): A batch of input data with keys `input_ids`, `attention_mask`, and `labels`.
            batch_idx (int): Index of the current batch.

        Logs:
            - Loss for the current batch during the test stage.
        """
        self._evaluate(batch, batch_idx, "test")
        # return {"test_loss": loss}

    def test_epoch_end(self, outputs):
        """
        Computes and logs test metrics at the end of the test epoch.

        Args:
            outputs (list): List of outputs from `test_step` (unused in this implementation).

        Logs:
            - Overall F1, precision, and recall for the entire test dataset.
            - F1, precision, and recall for specific entity types (PER, ORG, LOC).
        """
        # Calculate metrics over the entire test set
        metrics = self.metric.compute()

        #Overall results
        self.log("test_f1", metrics["overall_f1"], prog_bar=True)
        self.log("test_precision", metrics["overall_precision"], prog_bar=True)
        self.log("test_recall", metrics["overall_recall"], prog_bar=True)
        #Person
        try:
            self.log("PER_test_f1", metrics["PER"]["f1"], prog_bar=True)
            self.log("PER_test_precision", metrics["PER"]["precision"], prog_bar=True)
            self.log("PER_test_recall", metrics["PER"]["recall"], prog_bar=True)
        except KeyError:
            pass
        #Organization
        try:
            self.log("ORG_test_f1", metrics["ORG"]["f1"], prog_bar=True)
            self.log("ORG_test_precision", metrics["ORG"]["precision"], prog_bar=True)
            self.log("ORG_test_recall", metrics["ORG"]["recall"], prog_bar=True)
        except KeyError:
            pass
        #Location
        try:
            self.log("LOC_test_f1", metrics["LOC"]["f1"], prog_bar=True)
            self.log("LOC_test_precision", metrics["LOC"]["precision"], prog_bar=True)
            self.log("LOC_test_recall", metrics["LOC"]["recall"], prog_bar=True)
        except KeyError:
            pass

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            dict: A dictionary containing the optimizer and learning rate scheduler configuration.
        
        Optimizer:
            - AdamW optimizer is used for parameter updates with the specified learning rate.
        
        Learning Rate Scheduler:
            - A linear warmup and decay scheduler is configured using the HuggingFace `get_linear_schedule_with_warmup`.
            - Warmup is applied over the first 6% of total steps.
            - In federated mode, the scheduler is updated to match the current training round.
        """
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        # roberta used a linear warmup for first 6% steps over a total of 10 epochs (with early stopping)
        # we make sure to follow the same learning rate scheduling across the federated training
        total_number_of_epochs = 10
        warmup_share = 0.06
        total_number_of_steps = int(total_number_of_epochs*NUM_STEPS_PER_ROUND)
        num_warmup_steps = int(total_number_of_steps*warmup_share)

        lr_scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_number_of_steps
            )

        if self.mode == "federated":        
            # if doing federated training
            # update the train scheduler to the current round (starts from 1)
            for _ in range((self.current_round-1)*NUM_STEPS_PER_ROUND):
                lr_scheduler.step()
            
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1
            }
        }


def get_parameters(model):
    """
    Extracts the parameters of a PyTorch model and returns them as a list of NumPy arrays.

    Args:
        model (torch.nn.Module): The PyTorch model from which parameters are to be extracted.

    Returns:
        list: A list of NumPy arrays representing the parameters of the model.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters):
    """
    Updates the parameters of a PyTorch model using a provided list of NumPy arrays.

    Args:
        model (torch.nn.Module): The PyTorch model whose parameters are to be updated.
        parameters (list): A list of NumPy arrays representing the new parameter values.

    Modifies:
        The state dictionary of the model is updated with the new parameters.
    """
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def load_data(dataset_name, batch_size=32, num_workers=8, model_name="FacebookAI/xlm-roberta-large", cache_dir="/mimer/NOBACKUP/groups/naiss2024-22-1455/project-data/nerf_wasp/.cache"):
    """
    Loads, tokenizes, and prepares data for a specific NER dataset.

    Args:
        dataset_name (str): The name of the dataset to load.
        batch_size (int, optional): The batch size for the data loaders. Defaults to 32.
        num_workers (int, optional): The number of worker threads for loading data. Defaults to 8.
        model_name (str, optional): The name or path of the pretrained model tokenizer to use. Defaults to "FacebookAI/xlm-roberta-large".
        cache_dir (str, optional): Directory to cache the dataset and tokenizer. Defaults to a predefined path.

    Returns:
        tuple: A tuple containing:
            - train_loader (torch.utils.data.DataLoader): DataLoader for the training partition.
            - val_loader (torch.utils.data.DataLoader): DataLoader for the validation partition.
            - test_loader (torch.utils.data.DataLoader): DataLoader for the test partition.

    Notes:
        - This function assumes the dataset follows the "K2triinK/universal_ner_nordic_FL" format, with keys 
          such as "tokens" and "ner_tags".
        - The labels are aligned with tokenized inputs, and out-of-word tokens are assigned a label of -100.
        - Each partition is tokenized, formatted, and converted to PyTorch tensors for use in data loaders.
    """
    # Load the dataset
    partition = load_dataset("K2triinK/universal_ner_nordic_FL", dataset_name, cache_dir=cache_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize the dataset
    def tokenize_and_align_labels(batch):
        """
        Tokenize input text and align labels to the tokenized outputs.

        This function processes a batch of tokenized input data by:
        1. Tokenizing the input text (`tokens`) using a tokenizer configured for 
        handling word-level splits.
        2. Aligning the Named Entity Recognition (NER) labels (`ner_tags`) with 
        the tokenized outputs. For subword tokens or special tokens, it assigns 
        a label of `-100` to ignore them during training.

        Args:
            batch (dict): A batch of data containing:
                - `tokens` (list of list of str): The tokenized words for each example in the batch.
                - `ner_tags` (list of list of int): The corresponding NER labels for the tokens.

        Returns:
            dict: A dictionary containing:
                - `input_ids` (list of list of int): The tokenized input IDs.
                - `attention_mask` (list of list of int): The attention mask for the tokenized inputs.
                - `labels` (list of list of int): The aligned labels for each tokenized input, 
                with `-100` for tokens that should be ignored during training.

        Example:
            Input:
            batch = {
                "tokens": [["John", "lives", "in", "New", "York"]],
                "ner_tags": [[1, 0, 0, 3, 3]]
            }

            Output:
            {
                "input_ids": [[101, 2198, 3268, ..., 102]],
                "attention_mask": [[1, 1, 1, ..., 0]],
                "labels": [[1, -100, 0, 3, 3, -100, -100, ..., -100]]
            }

        Notes:
            - This function is designed to work with tokenizers that support 
            `is_split_into_words=True`, ensuring proper handling of pre-tokenized inputs.
            - Tokens corresponding to special tokens or padding are ignored in the `labels`.
        """
        # Tokenize the input text
        tokenized_inputs = tokenizer(batch["tokens"], truncation=True, padding="max_length", max_length=tokenizer.model_max_length, is_split_into_words=True)

        # Align labels with tokens
        labels = []
        for i, label in enumerate(batch["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            label_ids = [-100 if word_id is None else label[word_id] for word_id in word_ids]
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    # Apply tokenization to the entire dataset
    tokenized_train_partition = partition["train"].map(tokenize_and_align_labels, remove_columns=["idx", "text", "tokens", "ner_tags", "annotator"], batched=True)
    tokenized_train_partition.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    tokenized_val_partition = partition["validation"].map(tokenize_and_align_labels, remove_columns=["idx", "text", "tokens", "ner_tags", "annotator"], batched=True)
    tokenized_val_partition.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    tokenized_test_partition = partition["test"].map(tokenize_and_align_labels, remove_columns=["idx", "text", "tokens", "ner_tags", "annotator"], batched=True)
    tokenized_test_partition.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])

    # Attach dataset name to each partition
    tokenized_train_partition.name = dataset_name
    tokenized_val_partition.name = dataset_name
    tokenized_test_partition.name = dataset_name
    
    # Create data loaders for each split
    train_loader = DataLoader(
        tokenized_train_partition,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        tokenized_val_partition,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    test_loader = DataLoader(tokenized_test_partition, batch_size=batch_size, num_workers=num_workers)

    return train_loader, val_loader, test_loader