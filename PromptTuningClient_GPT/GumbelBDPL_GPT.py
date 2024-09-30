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
import torch.nn.functional as F

from preprocess_GPT import ApiCallLimitError, DOMAIN_DATASET, LABEL2ID_CONFIG, constrainScoreByWholeExact, CompleteGPT, create_batches # 1 import complete GPT. delete train_api_request. 


from scipy.optimize import minimize
import csv
import time

logger = logging.getLogger(__name__)


class ClientGumbelBDPL:
    def __init__(self, args, accelerator, client_partial_train_dataset, ngram_list, complete_GPT):

        # GPT
        self.complete_GPT = complete_GPT
        # 
        self.hingeloss = MarginLoss(margin=args.margin, target=False)
        self.ce_loss = CrossEntropyLoss()
        self.ngram_list = ngram_list

        # initialize prompt. 
        prompt_search_space = args.prompt_search_space
        prompt_length = args.prompt_length
        #self.prompts_probs = torch.FloatTensor([[1 / prompt_search_space] * prompt_search_space] * prompt_length)
        #self.prompts_probs.requires_grad = True
        # gumbel
        self.prompts_alpha = torch.FloatTensor([[1 / prompt_search_space] * prompt_search_space] * prompt_length)*0.01
        # prompts_alpha = torch.FloatTensor([[15.0] * prompt_search_space] * prompt_length)
        self.prompts_alpha.requires_grad = True
        self.prompts_probs = F.gumbel_softmax(torch.log(self.prompts_alpha), tau=args.tau)

        # 
        self.prompt_optimizer = AdamW([{
            "params": [self.prompts_alpha],   # optimize alpha. 
            "weight_decay": args.weight_decay,
        }], lr=args.prompt_learning_rate)

        # FL parameter. 
        self.num_local_step = args.num_client_local_step

        self.dataset = client_partial_train_dataset  # Local dataset for the client
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
        #self.prompts_probs.data = average_theta.clone().detach()
        #self.prompts_probs.requires_grad = True
        self.prompts_alpha.data = average_theta.clone().detach()
        self.prompts_alpha.requires_grad = True
        
        # Example training loop
        for _ in range(self.num_local_step):
            train_batches = create_batches(self.dataset, batch_size=args.per_device_train_batch_size, shuffle=True) # 2 change to add batch. 
            #train_batches = accelerator.prepare(train_batches)                                                       # 2 
            
            try:
                for step in range(len(train_batches['sentence'])):                                                      # 3 
                    self.prompts_alpha.data = torch.clamp(self.prompts_alpha.data, min=1e-15)
                    self.prompts_probs = F.gumbel_softmax(torch.log(self.prompts_alpha), tau=args.tau)
                    prompts_dist = torch.distributions.Categorical(self.prompts_probs)
                    with torch.no_grad():                                                                                     #4 All modified. 
                        if args.trial and step >= args.train_trial_step:
                            break
                        bsz = len(train_batches['sentence'][step])            # batch_size. 
                        labels = train_batches["labels"][step]
                        loss_list = []
                        prompts_discrete_indices_list = []
                        for k in range(args.sample_size):
                            prompts_discrete_indices = prompts_dist.sample() 
                            prompts_discrete_indices_list.append(prompts_discrete_indices) 

                            if args.use_ngram:
                                prompts_discrete_ngram_list = []
                                indices_list = prompts_discrete_indices.int().tolist()
                                for idx in indices_list:
                                    prompts_discrete_ngram_list.append(self.ngram_list[idx])
                                
                                prompts_discrete = ' '.join(prompts_discrete_ngram_list)
                            else: 
                                indices_list = prompts_discrete_indices.int().tolist()
                                prompts_discrete = tokenizer.decode(indices_list, clean_up_tokenization_spaces=False)

                            label_keys = list(self.label_to_id.keys())
                            converted_target = torch.tensor([self.label_to_id[label] for label in labels])

                            batch = []                                                                                 # 5 batch 的包装方式要重做。
                            label_probs = []
                            for i in range(len(train_batches['sentence'][step])): #  change to single one each. 
                                chat_obj = [{ "role":'user', "content" : prompts_discrete + '\t' + train_batches['sentence'][step][i] }]
                                label = train_batches['labels'][step][i]
                                response = self.complete_GPT.train_api_request(chat_obj, max_tokens=args.max_tokens, model_name=args.model_name_or_path, n=1, top_logprob=args.top_logprob)
                                labels_prob = self.complete_GPT.get_label_prob(response, chat_obj, label_keys, args)
                                batch.append(chat_obj)
                                #print(labels_prob)
                                label_probs.append(labels_prob) # if the prompt cannto get, it will be -10, meaning that it is very small. 

                                #print("the label is ", label)
                                #raise Exception()
                            
                            #label_probs = self.complete_GPT.get_regular_label_probs(responses, batch, label_keys, args, if_null = True)
                            logits = torch.stack(label_probs)   # logit 的结合方式要改。
                            pred = logits.argmax(dim=-1)

                            if args.ce_loss:
                                loss = self.ce_loss(logits, converted_target)
                            else:
                                loss = self.hingeloss(logits, converted_target)
                            loss_list.append(loss.item())

                            if self.complete_GPT.train_api_request.count >= args.api_limit:
                                raise ApiCallLimitError()

                        loss_avg = sum(loss_list) / args.sample_size
                        
                        self.prompt_optimizer.zero_grad()

                        # calculate the derivative w.r.t \alpha_{i,j} in Gumbel-softmax. 
                        derivative = (- self.prompts_probs / (self.prompts_alpha*args.tau)).repeat(args.sample_size, 1, 1)
                        for k, prompts_discrete_indices in enumerate(prompts_discrete_indices_list):
                            for i in range(args.prompt_length):  #
                                derivative[k][i][prompts_discrete_indices[i]] = (1-self.prompts_probs[i][prompts_discrete_indices[i]])/(self.prompts_alpha[i][prompts_discrete_indices[i]]*args.tau)   
                        

                        self.prompts_alpha.grad = torch.zeros_like(self.prompts_alpha)
                        for k in range(args.sample_size):
                            self.prompts_alpha.grad = self.prompts_alpha.grad + (1 / (args.sample_size - 1)) * (loss_list[k] - loss_avg) * derivative[k]
                        # Gumbel. 

                        torch.nn.utils.clip_grad_norm_(self.prompts_probs, 3)
                        
                        self.prompt_optimizer.step()
                        constrainScoreByWholeExact(self.prompts_probs)

                        if step % args.gradient_accumulation_steps == 0 or step == len(train_batches['sentence']) - 1:
                            self.completed_steps += 1
                        if self.completed_steps >= args.max_train_steps:
                            break

            except ApiCallLimitError:
                pass

        return self.prompts_probs.clone().detach()

    
    def evaluateGumbelBDPL(self, args, eval_batches, metric, ce_loss, config, accelerator, epoch, results, ngram_list, prompts_alpha=None, prompt_length=None,tokenizer=None):
        
        prompts_probs = F.gumbel_softmax(torch.log(prompts_alpha), tau=args.tau)
        if prompts_probs is not None:
            prompts_discrete_indices = prompts_probs.argmax(1)

            if args.use_ngram:
                prompts_discrete_ngram_list = []
                indices_list = prompts_discrete_indices.int().tolist()
                for idx in indices_list:
                    prompts_discrete_ngram_list.append(ngram_list[idx])
                prompts_discrete = ' '.join(prompts_discrete_ngram_list)

            else: 
                indices_list = prompts_discrete_indices.int().tolist()
                prompts_discrete = tokenizer.decode(indices_list, clean_up_tokenization_spaces=False)


        for step in range(len(eval_batches['sentence'])): # 200 batch , 每个batch 16个。

            print(f"evaluate step {step}")
            if args.trial and step >= args.eval_trial_step:
                break

            labels = eval_batches["labels"][step]

            # start
            label_keys = list(self.label_to_id.keys())
            converted_target = torch.tensor([self.label_to_id[label] for label in labels])
            
            batch = []                                                                                 # 5 batch 的包装方式要重做。
            label_probs = []
            for i in range(len(eval_batches['sentence'][step])): #  change to single one each. 

                chat_obj = [{ "role":'user', "content" : prompts_discrete + '\t' + eval_batches['sentence'][step][i] }]
                label = eval_batches['labels'][step][i]
                # 
                response = self.complete_GPT.train_api_request(chat_obj, max_tokens=args.max_tokens, model_name=args.model_name_or_path, n=1, top_logprob=args.top_logprob)
                labels_prob = self.complete_GPT.get_label_prob(response, chat_obj, label_keys, args)
                batch.append(chat_obj)
                label_probs.append(labels_prob) # if the prompt cannto get, it will be -10, meaning that it is very small. 
                
            
            #label_probs = self.complete_GPT.get_regular_label_probs(responses, batch, label_keys, args, if_null = True)
            logits = torch.stack(label_probs)   # logit 的结合方式要改。
            pred = logits.argmax(dim=-1)
            # end. 

            converted_target = converted_target[:len(logits)]  # 增加这个
            eval_loss_c = ce_loss(logits.view(-1, args.num_labels), converted_target)
            predictions = logits.argmax(dim=-1)

            if len(predictions.shape) == 0: predictions = predictions.unsqueeze(0)
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
        
        return eval_result, prompts_probs


    def testGumbelBDPL(self, args, test_batches, metric, accelerator, epoch, results, prompts_probs=None, prompt_length=None, tokenizer=None, linear_layer=None, prompts=None, label_to_id=None, test_batches_mm=None):
        
        if args.task_name == None or args.k_shot >= 0:
            if prompts_probs is not None:
                prompts_discrete_indices = prompts_probs.argmax(1)

                if args.use_ngram:
                    prompts_discrete_ngram_list = []
                    indices_list = prompts_discrete_indices.int().tolist()
                    for idx in indices_list:
                        prompts_discrete_ngram_list.append(self.ngram_list[idx])
                    prompts_discrete = ' '.join(prompts_discrete_ngram_list)

                else: 
                    indices_list = prompts_discrete_indices.int().tolist()
                    prompts_discrete = tokenizer.decode(indices_list, clean_up_tokenization_spaces=False)

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
                    chat_obj = [{ "role":'user', "content" : prompts_discrete + '\t' + test_batches['sentence'][step][i] }]
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
                    print(f"step is {step}")

                    labels = test_batches_mm['labels'][step]

                    # start
                    label_keys = list(self.label_to_id.keys())
                    converted_target = torch.tensor([self.label_to_id[label] for label in labels])
                    
                    batch = []                                                                                 # 5 batch 的包装方式要重做。
                    label_probs = []
                    for i in range(len(test_batches['sentence'][step])): #  change to single one each. 
                        chat_obj = [{ "role":'user', "content" : prompts_discrete + '\t' + test_batches['sentence'][step][i] }]
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