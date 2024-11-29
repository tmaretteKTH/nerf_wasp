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

disable_progress_bar()

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

class NERLightningModule(pl.LightningModule):
    
    def __init__(self, 
                 model_name="distilbert/distilbert-base-multilingual-cased", 
                 cache_dir="/mimer/NOBACKUP/groups/naiss2024-22-1455/project-data/nerf_wasp/.cache", 
                 num_labels=7, 
                 learning_rate=2e-5) -> None:
        super().__init__()
        self.model = AutoModelForTokenClassification.from_pretrained(model_name, cache_dir=cache_dir, num_labels=num_labels) 
        self.learning_rate = learning_rate
        self.metric = load_metric("seqeval", trust_remote_code=True)
        self.label_list = ["O", "B-PER", "I-PER", "B-ORG","I-ORG","B-LOC", "I-LOC"]
        
    def forward(self, input_ids, attention_mask, labels=None) -> Any:
        return self.model(input_ids, attention_mask=attention_mask, labels=labels)
    
    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs.loss
        self.log("train_loss", loss)
        return loss    
    
    def _evaluate(self, batch, batch_idx, stage=None):
        outputs = self(**batch)
        loss = outputs.loss
        logits = outputs.logits
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

        self.metric.add_batch(predictions=true_predictions, references=true_labels)
        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
    
    def validation_step(self, batch, batch_idx):
        self._evaluate(batch, batch_idx, "val")
        # return loss
        
    def validation_epoch_end(self, outputs):
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
        self._evaluate(batch, batch_idx, "test")
        # return {"test_loss": loss}

    def test_epoch_end(self, outputs):
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
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def load_data(dataset_name, batch_size=32, num_workers=8, model_name="FacebookAI/xlm-roberta-large", cache_dir="/mimer/NOBACKUP/groups/naiss2024-22-1455/project-data/nerf_wasp/.cache"):
    # Select the unique dataset for this client
    #partition = load_dataset('universalner/universal_ner', dataset_name, trust_remote_code=True)
    #'universalner/universal_ner' does not include Norwegian
    partition = load_dataset("K2triinK/universal_ner_nordic_FL", dataset_name, cache_dir=cache_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize the dataset
    def tokenize_and_align_labels(batch):
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