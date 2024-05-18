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


from preprocess import train_api_request, ApiCallLimitError, DOMAIN_DATASET, LABEL2ID_CONFIG


logger = logging.getLogger(__name__)


class ClientPromptTuning:
    def __init__(self, args, accelerator, model, client_partial_train_dataset, data_collator, config):
        self.hingeloss = MarginLoss(margin=args.margin, target=False)
        self.ce_loss = CrossEntropyLoss()
        self.config = config

        # optimizer. 
        prompt_parameters = [param for param in model.parameters() if param.requires_grad]
        self.prompt_optimizer = torch.optim.AdamW(prompt_parameters, lr=args.prompt_learning_rate)
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

        original_theta = model.prompt_encoder.default.embedding.weight.clone().detach()
        # model, assign the trainable parameter. 
        # train with local data. 
        for _ in range(self.num_local_step):
            try:
                for step, batch in enumerate(self.train_dataloader):
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

                    sequence_output = train_api_request(model, input_ids=input_ids, attention_mask=attention_mask) #
                    last_hidden_state = sequence_output[0].squeeze()
                    logits = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_pos]
                    logits = logits[:, interest_index]
                    #pred = logits.argmax(dim=-1)

                    if args.ce_loss:
                        loss = self.ce_loss(logits.view(-1, self.config.num_labels), converted_target)
                    else:
                        loss = self.hingeloss(logits, converted_target)

                    loss.backward()
                    self.prompt_optimizer.step()

                    if train_api_request.count >= args.api_limit:
                        raise ApiCallLimitError()

                    self.completed_steps += 1
                    if self.completed_steps >= args.max_client_train_steps:
                        break

            except ApiCallLimitError:
                pass
        # return the trained parameter.
        local_theta = model.prompt_encoder.default.embedding.weight.clone().detach()
        # Restore the model. 
        model.prompt_encoder.default.embedding.weight.data = original_theta

        return local_theta


def evaluatePromptTuning(args,  model, eval_dataloader, metric, ce_loss,config, accelerator, epoch, results, prompts_probs=None, prompt_length=None,tokenizer=None):
    model.prompt_encoder.default.embedding.weight.data.copy_(prompts_probs.data)

    for step, batch in enumerate(eval_dataloader):
        if args.trial and step >= 100:
            break
        bsz = len(batch['input_ids'])

        mask_pos=np.where(np.array(batch['input_ids'].cpu()) == tokenizer.mask_token_id) 
        mask_pos = torch.tensor(mask_pos[-1]) + args.prompt_length
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

def testPromptTuning(args, model, test_dataloader, metric, accelerator, epoch, results, prompts_probs=None, prompt_length=None, tokenizer=None, test_dataloader_mm=None):  
    if args.task_name == None or args.k_shot >= 0:
        model.prompt_encoder.default.embedding.weight.data.copy_(prompts_probs.data)

        for step, batch in enumerate(test_dataloader):
            if args.trial and step >= 100:
                break
            bsz = len(batch['input_ids'])

            mask_pos = np.where(np.array(batch['input_ids'].cpu()) == tokenizer.mask_token_id) 
            mask_pos = torch.tensor(mask_pos[-1]) + args.prompt_length
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

                mask_pos = np.where(np.array(batch['input_ids'].cpu()) == tokenizer.mask_token_id) 
                mask_pos = torch.tensor(mask_pos[-1]) + args.prompt_length
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
