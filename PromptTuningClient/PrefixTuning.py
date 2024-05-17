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

from scipy.optimize import minimize
import csv
import time

logger = logging.getLogger(__name__)

"""
格式：
updated_params = {
    'prefix_embeddings': <tensor>,
    'attention_params': [
        {
            'query': {
                'weight': <tensor>,  # Tensor with updated weights for the query in layer 0
                'bias': <tensor>     # Tensor with updated biases for the query in layer 0
            },
            'key': {
                'weight': <tensor>,  
                'bias': <tensor>    
            },
            'value': {
                'weight': <tensor>,  
                'bias': <tensor>    
            }
        },
        # Similar structures for other layers...
    ]
}

Example:
    params1 = ModelParams(torch.rand(768), [{'query': {'updated_weight': torch.rand(768), 'updated_bias': torch.rand(768)}}])
    params2 = ModelParams(torch.rand(768), [{'query': {'updated_weight': torch.rand(768), 'updated_bias': torch.rand(768)}}])

    result = params1 + params2
"""
class ModelParams:
    def __init__(self, prefix_embeddings, attention_params):
        self.prefix_embeddings = prefix_embeddings
        self.attention_params = attention_params

    def __add__(self, other):
        if not isinstance(other, ModelParams):
            raise ValueError("Can only add ModelParams with ModelParams")
        
        # Add prefix embeddings
        new_prefix_embeddings = self.prefix_embeddings + other.prefix_embeddings
        
        # Add attention parameters
        new_attention_params = []
        for self_layer, other_layer in zip(self.attention_params, other.attention_params):
            layer = {}
            for key in self_layer:
                layer[key] = {
                    'weight': self_layer[key]['weight'] + other_layer[key]['weight'],
                    'bias': self_layer[key]['bias'] + other_layer[key]['bias']
                }
            new_attention_params.append(layer)
        
        return ModelParams(new_prefix_embeddings, new_attention_params)
    
    def __radd__(self, other):
        # 允许 0 与 ModelParams 相加
        if other == 0:
            return self
        else:
            return self.__add__(other)
    
    def __mul__(self, scalar):
        if not isinstance(scalar, (int, float)):
            raise ValueError("Can only multiply ModelParams by an integer or a float")
        
        # Multiply prefix embeddings by scalar
        new_prefix_embeddings = self.prefix_embeddings * scalar
        
        # Multiply attention parameters by scalar
        new_attention_params = []
        for layer in self.attention_params:
            new_layer = {}
            for key in layer:
                new_layer[key] = {
                    'weight': layer[key]['weight'] * scalar,
                    'bias': layer[key]['bias'] * scalar
                }
            new_attention_params.append(new_layer)
        
        return ModelParams(new_prefix_embeddings, new_attention_params)
    
    def __rmul__(self, scalar):
        # This method handles multiplication when the scalar is on the left
        return self.__mul__(scalar)

    def element_size(self):
        # Call element_size() on the prefix_embeddings tensor
        return self.prefix_embeddings.element_size()

    def nelement(self):
        # Get total number of elements in prefix_embeddings
        total_elements = self.prefix_embeddings.nelement()
        
        # Get total number of elements in all weights and biases in attention_params
        for layer in self.attention_params:
            for param in layer.values():
                for x in param.values():
                    total_elements += x.nelement()       
        return total_elements
    
    def clone(self):
        # Clone prefix_embeddings
        cloned_prefix_embeddings = self.prefix_embeddings.clone()
        
        # Clone each layer in attention_params
        cloned_attention_params = []
        for layer in self.attention_params:
            cloned_layer = {}
            for key, param in layer.items():
                cloned_layer[key] = {
                    'weight': param['weight'].clone(),
                    'bias': param['bias'].clone()
                }
            cloned_attention_params.append(cloned_layer)
        
        return ModelParams(cloned_prefix_embeddings, cloned_attention_params)
    
    def detach(self):
        # Detach prefix_embeddings
        detached_prefix_embeddings = self.prefix_embeddings.detach()
        
        # Detach each layer in attention_params
        detached_attention_params = []
        for layer in self.attention_params:
            detached_layer = {}
            for key, param in layer.items():
                detached_layer[key] = {
                    'weight': param['weight'].detach(),
                    'bias': param['bias'].detach()
                }
            detached_attention_params.append(detached_layer)
        
        return ModelParams(detached_prefix_embeddings, detached_attention_params)

    def copy_from(self, source):
        """
        Copy the values from another ModelParams instance into this one.

        Parameters:
        - source (ModelParams): The source ModelParams instance to copy from.
        """
        if not isinstance(source, ModelParams):
            raise TypeError("Source must be an instance of ModelParams.")

        # Copy prefix embeddings
        self.prefix_embeddings.data.copy_(source.prefix_embeddings.data)

        # Copy attention parameters
        for dest_layer, src_layer in zip(self.attention_params, source.attention_params):
            for key in dest_layer:
                dest_layer[key]['weight'].data.copy_(src_layer[key]['weight'].data)
                dest_layer[key]['bias'].data.copy_(src_layer[key]['bias'].data)

class PrefixTunedRoberta(nn.Module):
    def __init__(self, args, model, config, prefix_length=10):
        super().__init__()
        self.config = config
        self.prefix_length = prefix_length
        self.prefix_embeddings = nn.Parameter(torch.randn(prefix_length, config.hidden_size))
        self.model = model
        # Freeze all parameters in the base model
        for param in self.model.parameters():
            param.requires_grad = False

        attention_params = []
        for layer in self.model.roberta.encoder.layer:
            attention = layer.attention.self
            layer_params = {}
            for component_name, component in zip(['query', 'key', 'value'], [attention.query, attention.key, attention.value]):
                # Enable training only for the specified prefix length
                component.weight[:prefix_length, :].requires_grad = True
                component.bias[:prefix_length].requires_grad = True
                    # Store the initial state of trainable parameters
                layer_params[component_name] = {
                    'weight': component.weight[:prefix_length, :],
                    'bias': component.bias[:prefix_length]
                }
            attention_params.append(layer_params)
            self.trainable_params = ModelParams(self.prefix_embeddings, attention_params)

        # label to id
        self.label_to_id = None
        if args.task_name:
            self.label_to_id = LABEL2ID_CONFIG[args.task_name]
        elif args.file_name:
            self.label_to_id = LABEL2ID_CONFIG[args.file_name]
                
        # has depenency outside. 
        if self.label_to_id is not None:
            self.config.label2id = self.label_to_id
            self.config.id2label = {id: label for label, id in config.label2id.items()}

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Generate prefix for each batch
        batch_size = input_ids.size(0)

        prefix = self.prefix_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)  # Shape: [batch_size, prefix_length, hidden_size]

        # Get embeddings from original model
        inputs_embeds = self.model.roberta.embeddings(input_ids)  # Assuming input_ids is not None

        # Concatenate prefix embeddings
        inputs_embeds = torch.cat((prefix, inputs_embeds), dim=1)

        # Adjust attention mask for the prefix
        if attention_mask is not None:
            prefix_attention_mask = torch.ones((batch_size, self.prefix_length), dtype=torch.long, device=input_ids.device)
            attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)

        # Pass to the original model's forward method
        output = self.model.forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)
        return output

class ClientPrefixTuning:
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

        # Backup the original trainable parameters
        original_params = ModelParams(model.prefix_embeddings.clone().detach(),
                                  [{component_name: {
                                      'weight': getattr(attention, component_name).weight.clone().detach()[:model.prefix_length, :],
                                      'bias': getattr(attention, component_name).bias.clone().detach()[:model.prefix_length]
                                    } for component_name in ['query', 'key', 'value']}
                                   for layer in model.model.roberta.encoder.layer
                                   for attention in [layer.attention.self]])

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

        # Collect the updated parameters
        updated_params = ModelParams(model.prefix_embeddings.clone().detach(),
                                    [{component_name: {
                                        'weight': getattr(attention, component_name).weight.clone().detach()[:model.prefix_length, :],
                                        'bias': getattr(attention, component_name).bias.clone().detach()[:model.prefix_length]
                                    } for component_name in ['query', 'key', 'value']}
                                    for layer in model.model.roberta.encoder.layer
                                    for attention in [layer.attention.self]])

        # Restore original parameters from backup
        model.prefix_embeddings.data = original_params.prefix_embeddings
        for layer, orig_layer_params in zip(model.model.roberta.encoder.layer, original_params.attention_params):
            attention = layer.attention.self
            for component_name in ['query', 'key', 'value']:
                getattr(attention, component_name).weight.data[:model.prefix_length, :] = orig_layer_params[component_name]['weight']
                getattr(attention, component_name).bias.data[:model.prefix_length] = orig_layer_params[component_name]['bias']

        return updated_params


def evaluatePrefixTuning(args, model, eval_dataloader, metric, ce_loss, config, accelerator, epoch, results, prompts_probs=None, tokenizer=None):
    model.eval()
    model.trainable_params.copy_from(prompts_probs)
    with torch.no_grad():
        for step, batch in enumerate(eval_dataloader):
            if args.trial and step >= 100:
                break

            input_ids = batch['input_ids']
            attention_mask = batch["attention_mask"]
            # Find the maks position. 
            # bsz = len(batch['input_ids'])
            mask_pos = np.where(np.array(input_ids.cpu()) == tokenizer.mask_token_id)     # 找到 mask position. 
            mask_pos = torch.tensor(mask_pos[-1]) 
                    
            # label and convert to target. 
            label = batch["labels"]
            label_keys = list(model.label_to_id.keys())
            label_map = {}
            for target in label_keys:
                label_map[tokenizer.encode(target, add_special_tokens=False)[0]] = model.label_to_id[target]

                converted_target = label.clone()
                for key, val in label_map.items():
                    converted_target[label == key] = val
                interest_index = list(label_map.keys())

                sequence_output = train_api_request(model, input_ids=input_ids, attention_mask=attention_mask) #
                last_hidden_state = sequence_output[0].squeeze()
                    
                logits = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_pos]
                logits = logits[:, interest_index]
                predictions = logits.argmax(dim=-1) 

            metric.add_batch(predictions=accelerator.gather(predictions), references=accelerator.gather(label))

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

def testPrefixTuning(args, model, test_dataloader, metric, accelerator, epoch, results, ngram_list, prompts_probs=None, prompt_length=None, tokenizer=None, test_dataloader_mm=None):
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