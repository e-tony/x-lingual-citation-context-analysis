from datasets import load_dataset
import pandas as pd
from pathlib import Path
import torch
import logging


class Dataset:
    def __init__(self, config):
        self.lowercase = True if "True" == config["PARAM"]["do_lower_case"] else False
        self.batch_size = int(config["PARAM"]["batch_size"])
        self.config = config

    def load(self, dataset_name):
        logging.info(f'Loading "{dataset_name}" dataset...')
        self.dataset = load_dataset(dataset_name)
        logging.info("Done loading.")

    def _encode(self, example):
        return self.tokenizer(
            example["string"],
            truncation=True,
            max_length=int(self.config["PARAM"]["max_len"]),
            padding="max_length",
        )

    def format(self, dataset):
        dataset = dataset.map(self._encode, batched=True)
        dataset.set_format(
            type="torch",
            columns=["input_ids", "token_type_ids", "attention_mask", "label"],
        )
        return dataset

    def format_data(self, tokenizer, batch_size=None):
        logging.info("Formatting data...")
        self.tokenizer = tokenizer
        if batch_size:
            self.batch_size = batch_size

        self.train_dataset = self.format(self.dataset["train"])
        self.validation_dataset = self.format(self.dataset["validation"])
        self.test_dataset = self.format(self.dataset["test"])
        logging.info("Done formatting.")
