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
from transformers.utils.versions import require_version
from transformers.models.roberta.configuration_roberta import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead, RobertaForMaskedLM
from transformers import RobertaModel
from torch.nn import CrossEntropyLoss
from loss import *
import wandb
from peft import get_peft_config, get_peft_model,  TaskType, PeftType
from peft import PromptTuningInit, PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig


from scipy.optimize import minimize
import csv
import time

logger = logging.getLogger(__name__)

from cmaes import CMA
import numpy as np

from preprocess import train_api_request, ApiCallLimitError, DOMAIN_DATASET, LABEL2ID_CONFIG

class ClientBBT:

    _A = None
    
    def __init__(self, args, accelerator, model, client_partial_train_dataset, data_collator, config):

        self.hingeloss = MarginLoss(margin=args.margin, target=False)
        self.ce_loss = CrossEntropyLoss()
        self.config = config

        # optimizer. 
        self.d = 20 # low dimension
        self.embedding_dim = model.get_input_embeddings().embedding_dim
        self.D = args.prompt_length * self.embedding_dim # prompt space dimension. 
        # Initialize the A matrix, this is the same across all clients. 
        if ClientBBT._A is None: 
            ClientBBT._A = torch.randn(self.D, self.d).to(args.device)  # the Mapping from low space to prompt space
        self.sigma = 1.2

        # cma_es
        self.cma_opts = {
            'seed': args.seed,
            'popsize': 1,
            'maxiter': 1,
            'verbose': -1,
            'bounds' : [-5, 5]
        }
        self.populiation_size = 2 # population size
        self.bounds = np.tile(np.array([-5, 5]), (self.d, 1)) # -5,5 bound for each element. 

        
        # FL parameter. 
        self.dataset = client_partial_train_dataset  # Local dataset for the client
        self.train_dataloader = DataLoader(self.dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
        self.train_dataloader = accelerator.prepare(self.train_dataloader)
        self.completed_steps = 0
    
        # label to id
        self.label_to_id = None
        if args.task_name:
            self.label_to_id = LABEL2ID_CONFIG[args.task_name]
        elif args.file_name:
            self.label_to_id = LABEL2ID_CONFIG[args.file_name]


    def get_len_dataset(self):
        return len(self.dataset)

    def local_training(self, args, model, tokenizer, average_theta, tracker):

        self.optimizer = CMA(mean=average_theta.detach().cpu().numpy(), sigma=self.sigma, bounds=self.bounds, population_size=self.populiation_size)
        #self.optimizer = cma.CMAEvolutionStrategy(np.zeros(self.d), self.sigma, inopts=self.cma_opts)
        # train with local data. 
        try:
            for step, batch in enumerate(self.train_dataloader):
                solutions = []
                fitness = []
                for _ in range(self.optimizer.population_size):  # population size.

                    # input_ids, attn_mask
                    input_ids = batch['input_ids']
                    attention_mask = batch["attention_mask"]
                    # Find the maks position. 
                    # bsz = len(batch['input_ids'])
                    mask_pos = np.where(np.array(input_ids.cpu()) == tokenizer.mask_token_id)     # 找到 mask position. 
                    mask_pos = torch.tensor(mask_pos[-1]) 
                    
                    # label and convert to target. 
                    label = batch["labels"]
                    label_keys = list(self.label_to_id.keys())
                    label_map = {}
                    for target in label_keys:
                        label_map[tokenizer.encode(target, add_special_tokens=False)[0]] = self.label_to_id[target]

                    converted_target = label.clone()
                    for key, val in label_map.items():
                        converted_target[label == key] = val
                    interest_index = list(label_map.keys())

                    # sample from z. cmaes
                    z = self.optimizer.ask()
                    z = torch.from_numpy(z).float().to(args.device)
                    batch_size = input_ids.size(0)
                    prefix_embedding = torch.matmul(ClientBBT._A, z)# p_0 is none
                    prefix_embedding = prefix_embedding.reshape(args.prompt_length, -1)
                    prefix = prefix_embedding.reshape((args.prompt_length, -1)).repeat(batch_size, 1, 1).to(args.device) # 这里有问题。需要的是
                    inputs_embeds = model.roberta.embeddings(input_ids)  # Assuming input_ids is not None
                    inputs_embeds = torch.cat((prefix, inputs_embeds), dim=1)    
                    prefix_attention_mask = torch.ones((batch_size, args.prompt_length), dtype=torch.long, device=input_ids.device)
                    attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

                    # forward with the embedding layer. 
                    sequence_output = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask) # model.forward
                    train_api_request.count += 1

                    last_hidden_state = sequence_output[0].squeeze()
                    logits = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_pos]
                    logits = logits[:, interest_index]
                    #pred = logits.argmax(dim=-1)

                    if args.ce_loss:
                        loss = self.ce_loss(logits.view(-1, self.config.num_labels), converted_target)
                    else:
                        loss = self.hingeloss(logits, converted_target)

                    # cma-es
                    solutions.append((z.cpu().detach().numpy(), loss))
                    
                    #solutions.append(z.cpu().numpy())
                    #fitness.append(loss)

                    if train_api_request.count >= args.api_limit:
                        raise ApiCallLimitError()

                    self.completed_steps += 1
                    if self.completed_steps >= args.max_client_train_steps:
                        break


                self.optimizer.tell(solutions)
                # self.optimizer.tell(solutions, fitness)

        except ApiCallLimitError:
            pass

        # return the trained parameter.
        best_z = min(solutions, key=lambda x: x[1])
        local_theta = torch.from_numpy(best_z[0]).to(args.device)
        return local_theta


    @classmethod
    def evaluateBBT(cls, args,  model, eval_dataloader, metric, ce_loss,config, accelerator, epoch, results, ngram_list, prompts_probs=None, prompt_length=None,tokenizer=None):

        for step, batch in enumerate(eval_dataloader):
            if args.trial and step >= 100:
                break
            bsz = len(batch['input_ids'])
            input_ids = batch['input_ids']
            attention_mask = batch["attention_mask"]

            mask_pos=np.where(np.array(batch['input_ids'].cpu()) == tokenizer.mask_token_id) 
            mask_pos = torch.tensor(mask_pos[-1])
            label_to_id = model.config.label2id 

            z = prompts_probs
            input_ids = batch['input_ids']
            batch_size = bsz
            prefix_embedding = torch.matmul(ClientBBT._A, z)# p_0 is none
            prefix_embedding = prefix_embedding.reshape(args.prompt_length, -1)
            prefix = prefix_embedding.reshape((args.prompt_length, -1)).repeat(batch_size, 1, 1).to(args.device) # 这里有问题。需要的是
            inputs_embeds = model.roberta.embeddings(input_ids)  # Assuming input_ids is not None
            inputs_embeds = torch.cat((prefix, inputs_embeds), dim=1)    
            prefix_attention_mask = torch.ones((batch_size, args.prompt_length), dtype=torch.long, device=input_ids.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

            # forward with the embedding layer. 
            sequence_output = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask) # model.forward
            
            last_hidden_state = sequence_output[0].squeeze()
            logits = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_pos]
            

            label = batch["labels"].to(args.device)
            label_keys = list(label_to_id.keys())
            label_map = {}
            for target in label_keys:
                label_map[tokenizer.encode(target, add_special_tokens=False)[0]] = label_to_id[target]
            converted_target = label.clone()
            for key, val in label_map.items():
                converted_target[label == key] = val
            interest_index = list(label_map.keys())
            logits = logits[:, interest_index]

            eval_loss_c = ce_loss(logits.view(-1, config.num_labels), converted_target)
            predictions = logits.argmax(dim=-1)

            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(converted_target),
            )

        if args.file_name in DOMAIN_DATASET:
            eval_metric = metric.compute(average='macro')
        else:
            eval_metric = metric.compute()

        logger.info("** eval **")
        logger.info(f"epoch {epoch}: {eval_metric}")

        if args.task_name == 'cola':
            key = 'matthews_correlation'
        elif args.task_name in ['mnli', 'sst2', 'wnli', 'rte', 'qnli'] or args.file_name in ['MR', 'CR']:
            key = 'accuracy'
        else:
            key = 'f1'

        eval_result = eval_metric[key]
        results.append(eval_result)

        return eval_result

    @classmethod
    def testBBT(cls, args, model, test_dataloader, metric, accelerator, epoch, results, ngram_list, prompts_probs=None, prompt_length=None, tokenizer=None, test_dataloader_mm=None):
        if args.task_name == None or args.k_shot >= 0:

            for step, batch in enumerate(test_dataloader):
                if args.trial and step >= 100:
                    break
                bsz = len(batch['input_ids'])
                input_ids = batch['input_ids']
                attention_mask = batch["attention_mask"]
                
                mask_pos = np.where(np.array(batch['input_ids'].cpu()) == tokenizer.mask_token_id) 
                mask_pos = torch.tensor(mask_pos[-1])
                label_to_id = model.config.label2id 
                
                # 
                z = prompts_probs
                input_ids = batch['input_ids']
                batch_size = bsz
                prefix_embedding = torch.matmul(ClientBBT._A, z)# p_0 is none
                prefix_embedding = prefix_embedding.reshape(args.prompt_length, -1)
                prefix = prefix_embedding.reshape((args.prompt_length, -1)).repeat(batch_size, 1, 1).to(args.device) # 这里有问题。需要的是
                inputs_embeds = model.roberta.embeddings(input_ids)  # Assuming input_ids is not None
                inputs_embeds = torch.cat((prefix, inputs_embeds), dim=1)    
                prefix_attention_mask = torch.ones((batch_size, args.prompt_length), dtype=torch.long, device=input_ids.device)
                attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
                sequence_output = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask) # model.forward
                   
                last_hidden_state = sequence_output[0].squeeze()
                logits = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_pos]

                label = batch["labels"].to(args.device)
                label_keys = list(label_to_id.keys())
                label_map = {}
                for target in label_keys:
                    label_map[tokenizer.encode(target, add_special_tokens=False)[0]]  = label_to_id[target]
                converted_target = label.clone()
                for key, val in label_map.items():
                    converted_target[label == key] = val
                interest_index = list(label_map.keys())
                logits = logits[:, interest_index]

                predictions = logits.argmax(dim=-1)
                metric.add_batch(
                    predictions=accelerator.gather(predictions),
                    references=accelerator.gather(converted_target),
                )
                    
            if args.file_name in DOMAIN_DATASET:
                test_metric = metric.compute(average='macro')
            else:
                test_metric = metric.compute()

            if args.task_name == 'mnli':
                for step, batch in enumerate(test_dataloader_mm):
                    bsz = len(batch['input_ids'])
                    
                    mask_pos = np.where(np.array(batch['input_ids'].cpu()) == tokenizer.mask_token_id) 
                    mask_pos = torch.tensor(mask_pos[-1])
                    label_to_id = model.config.label2id 

                    # 
                    z = prompts_probs
                    input_ids = batch['input_ids']
                    batch_size = bsz
                    prefix_embedding = torch.matmul(ClientBBT._A, z)# p_0 is none
                    prefix_embedding = prefix_embedding.reshape(args.prompt_length, -1)
                    prefix = prefix_embedding.reshape((args.prompt_length, -1)).repeat(batch_size, 1, 1).to(args.device) # 这里有问题。需要的是
                    inputs_embeds = model.roberta.embeddings(input_ids)  # Assuming input_ids is not None
                    inputs_embeds = torch.cat((prefix, inputs_embeds), dim=1)    
                    prefix_attention_mask = torch.ones((batch_size, args.prompt_length), dtype=torch.long, device=input_ids.device)
                    attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
                    sequence_output = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask) # model.forward

                    last_hidden_state = sequence_output[0].squeeze()
                    logits = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_pos]

                    label = batch["labels"].to(args.device)
                    label_keys = list(label_to_id.keys())
                    label_map = {}
                    for target in label_keys:
                        label_map[tokenizer.encode(target, add_special_tokens=False)[0]]  = label_to_id[target]
                    converted_target = label.clone()
                    for key, val in label_map.items():
                        converted_target[label == key] = val
                    interest_index = list(label_map.keys())
                    logits = logits[:, interest_index]

                    predictions = logits.argmax(dim=-1)
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