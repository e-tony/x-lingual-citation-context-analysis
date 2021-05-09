from argparse import ArgumentParser
from pathlib import Path
import argparse
import configparser
import logging
import pandas as pd
import random
import datasets

from model import CitationClassificationModel
from reader import Dataset
from helper import init_logger, make_output_dir


class CitationClassifier:
    def __init__(self, args, config, hp_search=None):
        self.args = args
        self.config = config
        self.hp_search = hp_search

        self.model_name = config["PARAM"]["model"]
        self.dataset_name = config["CONFIG"]["dataset"]
        self.output_dir = make_output_dir(args, config, self.model_name)

        self.trained_model_path = None

        # start logging
        init_logger(args, config, self.output_dir)

        self.load_model()

    def prepare_data(self):
        logging.info("Preparing data...")

        if self.dataset_name == "Athar":
            path = Path(
                "../data/citation_sentiment/citation_sentiment_corpus_cleaned.tsv"
            )
            train_path = Path(
                "../data/citation_sentiment/citation_sentiment_corpus_cleaned_train.tsv"
            )
            val_path = Path(
                "../data/citation_sentiment/citation_sentiment_corpus_cleaned_val.tsv"
            )
            test_path = Path(
                "../data/citation_sentiment/citation_sentiment_corpus_cleaned_test.tsv"
            )
            names = ["source_paper", "target_paper", "label", "string"]

            if path.is_file():
                if (
                    not train_path.is_file()
                    or not val_path.is_file()
                    or not test_path.is_file()
                ):
                    df = pd.read_csv(path, sep="\t", names=names)

                    idxs = list(range(df.shape[0]))
                    random.shuffle(idxs)

                    _train_size = 0.8
                    _train_idx = int(len(idxs) * _train_size)
                    _val_size = 0.1
                    _val_idx = int(len(idxs) * _val_size) + 1

                    train_idxs = idxs[:_train_idx]
                    val_idxs = idxs[_train_idx : _train_idx + _val_idx]
                    test_idxs = idxs[_train_idx + _val_idx :]

                    df_train = df.iloc[train_idxs]
                    df_val = df.iloc[val_idxs]
                    df_test = df.iloc[test_idxs]

                    df_train.to_csv(
                        str(train_path), sep="\t", index=False, header=False
                    )
                    df_val.to_csv(str(val_path), sep="\t", index=False, header=False)
                    df_test.to_csv(str(test_path), sep="\t", index=False, header=False)
                else:
                    df_train = pd.read_csv(train_path, sep="\t", names=names)
                    df_val = pd.read_csv(val_path, sep="\t", names=names)
                    df_test = pd.read_csv(test_path, sep="\t", names=names)

                df_train.label = df_train.label.replace({"o": 0, "p": 1, "n": 2})
                df_val.label = df_val.label.replace({"o": 0, "p": 1, "n": 2})
                df_test.label = df_test.label.replace({"o": 0, "p": 1, "n": 2})

                train_dataset = datasets.Dataset.from_pandas(df_train)
                val_dataset = datasets.Dataset.from_pandas(df_val)
                test_dataset = datasets.Dataset.from_pandas(df_test)

                dataset_dict = datasets.DatasetDict(
                    {
                        "train": train_dataset,
                        "validation": val_dataset,
                        "test": test_dataset,
                    }
                )

                self.dataset = Dataset(self.config)
                self.dataset.dataset = dataset_dict
                self.dataset.format_data(self.classifier.tokenizer)
            else:
                raise Exception(
                    f"Cannot find dataset {self.dataset_name} with path: {str(path)}"
                )
        else:
            self.dataset = Dataset(self.config)
            self.dataset.load(self.dataset_name)
            self.dataset.format_data(self.classifier.tokenizer)

        self.data = {}
        self.data["train"] = self.dataset.train_dataset
        self.data["val"] = self.dataset.validation_dataset
        self.data["test"] = self.dataset.test_dataset

        logging.info("Done preparing.")

    def load_model(self, model_path=None):
        model_name = self.model_name
        if model_path:
            model_name = model_path
        self.classifier = CitationClassificationModel(
            model_name, self.config, output_dir=self.output_dir, debug=self.args.debug
        )

    def train(self):
        self.classifier.train(self.data["train"], self.data["val"], self.hp_search)
        self.trained_model_path = self.output_dir / "pytorch_model.bin"

    def eval(self, model_path=None):
        self.classifier.predict(
            dataset=self.dataset.dataset["test"], model_path=model_path
        )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="./config.ini", type=str, required=True)
    parser.add_argument("--do-train", required=False, action="store_true")
    parser.add_argument("--do-eval", required=False, action="store_true")
    parser.add_argument("--eval-model", type=str, required=False)
    parser.add_argument("--hp-search", required=False, action="store_true")
    parser.add_argument("--debug", required=False, action="store_true")
    args = parser.parse_args()

    if Path(args.config).is_file():
        config = configparser.ConfigParser()
        config.read(args.config)
    else:
        raise Exception(f"Config path is not a file: {Path(args.config).absolute()}")

    trainer = CitationClassifier(args, config)
    trainer.prepare_data()

    if args.do_train:
        trainer.train()

    if args.do_eval:
        trainer.eval(args.eval_model)
