import argparse
import logging
import torch
import math
import os
import random
import datasets
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from accelerate import Accelerator
from torch.optim import Adam, AdamW
from transformers import (
    AutoTokenizer,
    DataCollatorWithPadding,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
)
from accelerate import Accelerator
from transformers.utils.versions import require_version
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead, RobertaForMaskedLM
from transformers import RobertaModel
from torch.nn import CrossEntropyLoss
from loss import *
import wandb
from peft import get_peft_config, get_peft_model,  TaskType, PeftType
from peft import PromptTuningInit, PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig

from preprocess_GPT import ApiCallLimitError, DOMAIN_DATASET, LABEL2ID_CONFIG, constrainScoreByWholeExact, CompleteGPT, create_batches # 1 import complete GPT. delete train_api_request. 


from scipy.optimize import minimize
import csv
import time

logger = logging.getLogger(__name__)




class ClientNoPrompt:
    def __init__(self, args, accelerator, client_partial_train_dataset, ngram_list, complete_GPT):

        # GPT
        self.complete_GPT = complete_GPT
        # 
        self.hingeloss = MarginLoss(margin=args.margin, target=False)
        self.ce_loss = CrossEntropyLoss()
        self.ngram_list = ngram_list


        self.dataset = client_partial_train_dataset  # Local dataset for the client
        self.completed_steps = 0

        # label to id
        self.label_to_id = None
        if args.task_name:
            self.label_to_id = LABEL2ID_CONFIG[args.task_name]
        elif args.file_name:
            self.label_to_id = LABEL2ID_CONFIG[args.file_name]

    def testNoPrompt(self, args, test_batches, metric, accelerator, epoch, results, tokenizer=None, linear_layer=None, prompts=None, label_to_id=None, test_batches_mm=None):
        if args.task_name == None or args.k_shot >= 0:

            for step in range(len(test_batches['sentence'])):
                if args.trial and step >= args.test_trial_step:
                    break
                labels = test_batches['labels'][step]

                # start
                label_keys = list(self.label_to_id.keys())
                converted_target = torch.tensor([self.label_to_id[label] for label in labels])
                
                batch = []                                                                                 # 5 batch 的包装方式要重做。
                label_probs = []
                for i in range(len(test_batches['sentence'][step])): #  change to single one each. 
                    chat_obj = [{ "role":'user', "content" : test_batches['sentence'][step][i] }]
                    label = test_batches['labels'][step][i]
                    # 
                    response = self.complete_GPT.train_api_request(chat_obj, max_tokens=args.max_tokens, model_name=args.model_name_or_path, n=1, top_logprob=args.top_logprob)
                    labels_prob = self.complete_GPT.get_label_prob(response, chat_obj, label_keys, args)
                    batch.append(chat_obj)
                    print(labels_prob)   
                    label_probs.append(labels_prob) # if the prompt cannto get, it will be -10, meaning that it is very small. 
                
                #label_probs = self.complete_GPT.get_regular_label_probs(responses, batch, label_keys, args, if_null = True)
                logits = torch.stack(label_probs)   # logit 的结合方式要改。
                
                # end. 
                predictions = logits.argmax(dim=-1)

                if len(predictions.shape) == 0: predictions = predictions.unsqueeze(0)
                
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(converted_target),
                )

            if args.file_name in DOMAIN_DATASET:
                test_metric = metric.compute(average='macro')
            else:
                test_metric = metric.compute()

            if args.task_name == 'mnli':
                for step in range(len(test_batches_mm['sentence'])):
                    if args.trial and step >= args.test_trial_step:
                        break

                    labels = test_batches_mm['labels'][step]

                    # start
                    label_keys = list(self.label_to_id.keys())
                    converted_target = torch.tensor([self.label_to_id[label] for label in labels])
                    
                    batch = []                                                                                 # 5 batch 的包装方式要重做。
                    label_probs = []
                    for i in range(len(test_batches['sentence'][step])): #  change to single one each. 
                        chat_obj = [{ "role":'user', "content" : test_batches['sentence'][step][i] }]
                        label = test_batches['labels'][step][i]
                        # 
                        response = self.complete_GPT.train_api_request(chat_obj, max_tokens=args.max_tokens, model_name=args.model_name_or_path, n=1, top_logprob=args.top_logprob)
                        labels_prob = self.complete_GPT.get_label_prob(response, chat_obj, label_keys, args)
                        batch.append(chat_obj)
                        label_probs.append(labels_prob) # if the prompt cannto get, it will be -10, meaning that it is very small. 
                    
                    #label_probs = self.complete_GPT.get_regular_label_probs(responses, batch, label_keys, args, if_null = True)
                    logits = torch.stack(label_probs)   # logit 的结合方式要改。
                    predictions = logits.argmax(dim=-1)
                    # end. 
                    if len(predictions.shape) == 0: predictions = predictions.unsqueeze(0)
                    
                    metric.add_batch(
                        predictions=accelerator.gather(predictions),
                        references=accelerator.gather(converted_target),
                    )
                    test_metric_mm = metric.compute()

            if args.task_name == 'cola':
                key = 'matthews_correlation'
            elif args.task_name in ['mnli', 'sst2', 'wnli', 'rte', 'qnli'] or args.file_name in ['MR', 'CR']:
                key = 'accuracy'
            else:
                key = 'f1'
            test_result = test_metric[key]
            results.append(test_result)

            logger.info("** test **")
            logger.info(f"epoch {epoch}: {test_metric}")
            if args.use_wandb:
                for key in test_metric.keys():
                    eval_key = 'Black_test_' + key
                    wandb.log({eval_key: test_metric[key]})
                if args.task_name == 'mnli':
                    for key in test_metric_mm.keys():
                        eval_key = 'Black_test_' + key + '_mm'
                        wandb.log({eval_key: test_metric_mm[key]})