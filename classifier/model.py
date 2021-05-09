import torch
from torch.utils.data import DataLoader
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune.schedulers import PopulationBasedTraining
from transformers import AutoConfig, AutoModelForSequenceClassification
from transformers import AlbertTokenizerFast, BertTokenizerFast, DistilBertTokenizerFast, ElectraTokenizerFast, GPT2TokenizerFast, ReformerTokenizerFast, RobertaTokenizerFast, T5TokenizerFast, XLMRobertaTokenizerFast, XLNetTokenizerFast
from transformers import Trainer, TrainingArguments
from transformers import TextClassificationPipeline
from transformers import get_linear_schedule_with_warmup
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import time
import datetime
from pathlib import Path
import logging
from sklearn.metrics import classification_report
from metrics import compute_metrics_sklearn, compute_metrics_torch


class CitationClassificationModel:
    def __init__(self, model_name, config, output_dir=None, debug=False):
        self.debug = debug
        self.config = config
        self.output_dir = output_dir if output_dir else config['CONFIG']['output_dir']
        self.logging_dir = Path(self.output_dir) / 'logs'

        if not self.logging_dir.is_dir():
            self.logging_dir.mkdir(parents=True)

        # Model parameters
        self.model_name = config['PARAM']['model']
        self.do_lower_case = True if self.config['PARAM']['do_lower_case'] == 'True' else False
        # self.use_bfloat16 = True if self.config['PARAM']['bfloat16'] == 'True' else False

        logging.info('Loading pre-trained tokenizer...')
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name)

        self.tokenizer.do_lower_case = self.do_lower_case
        self.tokenizer.model_max_length = 512 
        logging.info('Done.')

        self.model_config = AutoConfig.from_pretrained(self.model_name)
        self.update_config()

        logging.info('Loading pre-trained model...')
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.model_config)
        logging.info(f'Loaded model from path: {self.model_name}')

    def model_init(self):
        return AutoModelForSequenceClassification.from_pretrained(self.model_name, config=self.model_config)
    
    def update_config(self):
        self.model_config.num_labels = int(self.config['CONFIG']['labels'])

    def init_training(self):
        # Training hyperparameters
        self.epochs = int(self.config['PARAM']['epochs'])
        self.batch_size = int(self.config['PARAM']['batch_size'])
        self.warmup_steps = int(self.config['PARAM']['warmup_steps'])
        self.weight_decay = float(self.config['PARAM']['weight_decay'])
        self.max_len = int(self.config['PARAM']['max_len'])

    def save_model(self, save_path: Path):
        if not save_path.is_dir():
            save_path.mkdir(parents=True)
        torch.save(self.model, save_path)
        logging.info(f'Saved model to path: {save_path}')

    def load_model(self, path):
        if Path(path).is_file():
            logging.info('Loading local model state dict...')
            self.model.load_state_dict(torch.load(path))
            logging.info(f'Loaded model from path: {path}')
        else:
            logging.error(f'Model path does not exist: {path}')
            raise Exception(f'The specified file path ({path}) does not exist!')

    def train(self, train_dataset, eval_dataset, hp_search):

        logging.info('Initializing trainer...')
        self.init_training()

        training_args = TrainingArguments(
            output_dir=self.output_dir,        
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size*4,
            warmup_steps=self.warmup_steps,
            weight_decay=self.weight_decay,
            logging_dir=self.logging_dir,
            # logging_strategy='steps',
            # logging_steps=500,
            evaluation_strategy='steps',
            eval_steps=500,
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            compute_metrics=compute_metrics_sklearn,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            # model_init=self.model_init,
        )
        logging.info('Trainer initialized.')

        if hp_search:
            logging.info('Conducting hyperparameter search...')
            def hp_space_fn():
                config = {
                        "warmup_steps": tune.choice([50, 100, 500, 1000]),
                        "learning_rate": tune.choice([1.5e-5, 2e-5, 3e-5, 4e-5]),
                        "num_train_epochs": tune.quniform(0.0, 10.0, 0.5),
                }
                return config

            trainer.hyperparameter_search(
                direction='maximize',
                backend='ray',
                scheduler=PopulationBasedTraining(
                    time_attr='time_total_s',
                    metric='eval_f1_thr_0',
                    mode='max',
                    perturbation_interval=600.0,
                ),
                hp_space=hp_space_fn,
                loggers=DEFAULT_LOGGERS,
            )
        else:
            logging.info('Training...')
            trainer.train()
            logging.info('Done training.')
            logging.info('Evaluating...')
            res = trainer.evaluate()
            logging.info('Validation results:')
            logging.info(f'Eval loss: {res["eval_loss"]}, Eval acc: {res["eval_accuracy"]}, Eval F1: {res["eval_f1"]}, Eval P: {res["eval_precision"]}, Eval R: {res["eval_recall"]}')
            logging.info('Done evaluating.')

            trainer.save_model()
            trainer.save_state()
            self.trained_model_path = self.output_dir / 'pytorch_model.bin'
            assert (self.trained_model_path.is_file())
            logging.info(f'Saved model to path: {self.output_dir}')

    def predict(self, dataset, model_path=None, get_performance=True):

        logging.info('Testing model...')
        if model_path:
            path = model_path
        else:
            path = self.trained_model_path
        self.load_model(path)

        classifier = TextClassificationPipeline(model=self.model, tokenizer=self.tokenizer, return_all_scores=True, device=0)
        dataloader = DataLoader(dataset, batch_size=16)

        true = []
        _pred = []

        with torch.no_grad():
            for batch in tqdm(dataloader):
                batch['labels'] = batch.pop('label')
                outputs = classifier(batch['string'], truncation=True)

                true.append(batch['labels'])
                _pred.append(outputs)

        true = torch.tensor([v for l in true for v in l])
        pred = [v for l in _pred for v in l]
        pred = torch.tensor([[torch.tensor(l['score']) for l in p] for p in pred])

        num_classes = int(self.config['CONFIG']['labels'])
        acc, p, r, f1 = compute_metrics_torch(pred, true, num_classes)
        logging.info('Performance stats:')
        logging.info(f'Acc: {acc}, P: {p}, R: {r}, F1: {f1}')

        dataset_name = self.config['CONFIG']['dataset']
        target_names = []
        if dataset_name == 'scicite':
            target_names = ['Method', 'Background', 'Result']
        elif dataset_name == 'Athar':
            target_names = ['Neutral', 'Positive', 'Negative']

        if target_names:
            report = classification_report(true.tolist(), torch.argmax(pred, axis=1).tolist(), target_names=target_names)
            logging.info(f'Classification report: \n{str(report)}')

        logging.info('Done testing.')
