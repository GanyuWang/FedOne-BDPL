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

from preprocess import train_api_request, ApiCallLimitError, DOMAIN_DATASET, LABEL2ID_CONFIG, constrainScoreByWholeExact


from scipy.optimize import minimize
import csv
import time

logger = logging.getLogger(__name__)


class ClientBDPL:
    def __init__(self, args, accelerator, client_partial_train_dataset, data_collator, config, ngram_list):

        self.hingeloss = MarginLoss(margin=args.margin, target=False)
        self.ce_loss = CrossEntropyLoss()
        self.config = config
        self.ngram_list = ngram_list

        # initialize prompt. 
        prompt_search_space = args.prompt_search_space
        prompt_length = args.prompt_length
        self.prompts_probs = torch.FloatTensor([[1 / prompt_search_space] * prompt_search_space] * prompt_length)
        self.prompts_probs.requires_grad = True
        # 
        self.prompt_optimizer = AdamW([{
            "params": [self.prompts_probs],
            "weight_decay": args.weight_decay,
        }], lr=args.prompt_learning_rate)

        # FL parameter. 
        self.num_local_step = args.num_client_local_step

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

        # Load the average prompt into the client's model
        self.prompts_probs.data = average_theta.clone().detach()
        self.prompts_probs.requires_grad = True
        
        # Example training loop
        for _ in range(self.num_local_step):
            try:
                for step, batch in enumerate(self.train_dataloader):
                    prompts_dist = torch.distributions.Categorical(self.prompts_probs)
                    with torch.no_grad():
                        if args.trial and self.completed_steps >= 100:
                            break
                        bsz = len(batch['input_ids'])             # batch_size. 
                        label = batch["labels"].to(args.device)
                        loss_list = []
                        prompts_discrete_indices_list = []
                        for k in range(args.sample_size):
                            prompts_discrete_indices = prompts_dist.sample() 
                            prompts_discrete_indices_list.append(prompts_discrete_indices) 
                            if args.use_ngram:
                                prompts_discrete_indices_ngram_list = []
                                indices_list = prompts_discrete_indices.int().tolist() # 采样的 index. 
                                for idx in indices_list:
                                    prompts_discrete_indices_ngram_list.append(self.ngram_list[idx]) 
                                prompts_discrete_ngram_indices = torch.tensor(prompts_discrete_indices_ngram_list)
                                cur_input_ids = torch.cat([torch.zeros(bsz, 1, dtype=torch.long).to(args.device), prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)
                            else: 
                                cur_input_ids = torch.cat([torch.zeros(bsz, 1, dtype=torch.long).to(args.device), prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1) # CLS + Discrete Prompt + input_ids

                            cur_attention_mask = torch.cat([torch.ones(bsz, 1).to(args.device), torch.ones(bsz, args.prompt_length).to(args.device), batch["attention_mask"][:, 1:]],dim=1) # [0, 1(prompt length), original_attention_mask]
                            mask_pos = np.where(np.array(cur_input_ids.cpu()) == tokenizer.mask_token_id)     # 找到 mask position. 
                            mask_pos = torch.tensor(mask_pos[-1]) 
                            sequence_output = train_api_request(model, input_ids=cur_input_ids, attention_mask=cur_attention_mask)

                            last_hidden_state = sequence_output[0].squeeze()
                            logits = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_pos]

                            label_keys = list(self.label_to_id.keys())
                            label_map = {}
                            for target in label_keys:
                                label_map[tokenizer.encode(target, add_special_tokens=False)[0]] = self.label_to_id[target]
                            
                            converted_target = label.clone()
                            for key, val in label_map.items():
                                converted_target[label == key] = val
                            interest_index = list(label_map.keys())
                            logits = logits[:, interest_index]
                            pred = logits.argmax(dim=-1)

                            if args.ce_loss:
                                loss = self.ce_loss(logits.view(-1, self.config.num_labels), converted_target)
                            else:
                                loss = self.hingeloss(logits, converted_target)
                            loss_list.append(loss.item())

                            if train_api_request.count >= args.api_limit:
                                raise ApiCallLimitError()

                        loss_avg = sum(loss_list) / args.sample_size
                        
                        self.prompt_optimizer.zero_grad()

                        if args.bdpl_gradient_method == "negative": #一个正其他负。
                            derivative = (-1 / self.prompts_probs).repeat(args.sample_size, 1, 1)
                            for k, prompts_discrete_indices in enumerate(prompts_discrete_indices_list):
                                for i in range(args.prompt_length):
                                        derivative[k][i][prompts_discrete_indices[i]] *= -1  # 只有一个正。其他负。
                        elif args.bdpl_gradient_method == "zero": # 一个正其他0.
                            derivative_1devided_by_p = (1 / self.prompts_probs).repeat(args.sample_size, 1, 1)
                            derivative = (torch.zeros_like(self.prompts_probs)).repeat(args.sample_size, 1, 1)
                            for k, prompts_discrete_indices in enumerate(prompts_discrete_indices_list):
                                for i in range(args.prompt_length):
                                        derivative[k][i][prompts_discrete_indices[i]] = derivative_1devided_by_p[k][i][prompts_discrete_indices[i]]  # 只有一个正。其他0。
                        elif args.bdpl_gradient_method == "normalize": # 一个正 其他 - 1/prompt_searching_space * p. 
                            derivative = (-1 / self.prompts_probs /args.prompt_search_space).repeat(args.sample_size, 1, 1)
                            for k, prompts_discrete_indices in enumerate(prompts_discrete_indices_list):
                                for i in range(args.prompt_length):
                                        derivative[k][i][prompts_discrete_indices[i]] *= -1 * args.prompt_search_space  # 只有一个正 1/p。其他 - 1/p /searching space.
                        else:
                            raise Exception("No bdpl_gradient calcualtion selected. ")
                                

                        self.prompts_probs.grad = torch.zeros_like(self.prompts_probs)
                        for k in range(args.sample_size):
                            self.prompts_probs.grad += 1 / (args.sample_size - 1) * (loss_list[k] - loss_avg) * derivative[k]

                        torch.nn.utils.clip_grad_norm_(self.prompts_probs, 3)
                        
                        self.prompt_optimizer.step()
                        constrainScoreByWholeExact(self.prompts_probs)

                        self.completed_steps += 1
                        if self.completed_steps >= args.max_client_train_steps:
                            break

            except ApiCallLimitError:
                pass

        return self.prompts_probs.clone().detach()


def evaluateBDPL(args,  model, eval_dataloader, metric, ce_loss,config, accelerator, epoch, results, ngram_list, prompts_probs=None, prompt_length=None,tokenizer=None):
    prompts_discrete_indices = prompts_probs.argmax(1)

    if args.use_ngram:
        prompts_discrete_indices_ngram_list = []
        indices_list = prompts_discrete_indices.int().tolist()
        for idx in indices_list:
            prompts_discrete_indices_ngram_list.append(ngram_list[idx])
        prompts_discrete_ngram_indices = torch.tensor(prompts_discrete_indices_ngram_list)

    for step, batch in enumerate(eval_dataloader):
        if args.trial and step >= 100:
            break
        bsz = len(batch['input_ids'])

        if args.use_ngram:
            batch['input_ids'] = torch.cat([torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)
        else:
            batch['input_ids'] = torch.cat([torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)
        batch["attention_mask"] = torch.cat([torch.ones(bsz, prompt_length).to(args.device), batch["attention_mask"]],dim=1)

        mask_pos=np.where(np.array(batch['input_ids'].cpu()) == tokenizer.mask_token_id) 
        mask_pos = torch.tensor(mask_pos[-1])
        label_to_id = model.config.label2id 

        sequence_output = model(input_ids=batch['input_ids'], attention_mask=batch["attention_mask"])
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

def testBDPL(args, model, test_dataloader, metric, accelerator, epoch, results, ngram_list, prompts_probs=None, prompt_length=None, tokenizer=None, test_dataloader_mm=None):
    if args.task_name == None or args.k_shot >= 0:
        prompts_discrete_indices = prompts_probs.argmax(1)
        #raise Exception(prompts_discrete_indices)

        if args.use_ngram:
            prompts_discrete_indices_ngram_list = []
            indices_list = prompts_discrete_indices.int().tolist()
            for idx in indices_list:
                prompts_discrete_indices_ngram_list.append(ngram_list[idx])
            prompts_discrete_ngram_indices = torch.tensor(prompts_discrete_indices_ngram_list)

        for step, batch in enumerate(test_dataloader):
            if args.trial and step >= 100:
                break
            bsz = len(batch['input_ids'])
            
            if args.use_ngram:
                batch['input_ids'] = torch.cat([torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)
                prompt_sample = tokenizer.decode(prompts_discrete_indices_ngram_list)
            else:
                batch['input_ids'] = torch.cat([torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)
            batch["attention_mask"] = torch.cat([torch.ones(bsz, prompt_length).to(args.device), batch["attention_mask"]],dim=1)

            mask_pos = np.where(np.array(batch['input_ids'].cpu()) == tokenizer.mask_token_id) 
            mask_pos = torch.tensor(mask_pos[-1])
            label_to_id = model.config.label2id 
            sequence_output = model(input_ids=batch['input_ids'], attention_mask=batch["attention_mask"])
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
                
                if args.use_ngram:
                    batch['input_ids'] = torch.cat([torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)
                    prompt_sample = tokenizer.decode(prompts_discrete_indices_ngram_list)
                else:
                    batch['input_ids'] = torch.cat([torch.zeros(bsz,1, dtype=torch.long).to(args.device), prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)
                batch["attention_mask"] = torch.cat([torch.ones(bsz, prompt_length).to(args.device), batch["attention_mask"]],dim=1)

                mask_pos = np.where(np.array(batch['input_ids'].cpu()) == tokenizer.mask_token_id) 
                mask_pos = torch.tensor(mask_pos[-1])
                label_to_id = model.config.label2id 
                sequence_output = model(input_ids=batch['input_ids'], attention_mask=batch["attention_mask"])
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

        return test_result