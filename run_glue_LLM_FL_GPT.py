import argparse
import logging
import torch
import math
import os
import random
import pandas as pd
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
#from transformers.models.roberta.configuration_roberta import RobertaConfig
#from transformers.models.roberta.modeling_roberta import RobertaClassificationHead, RobertaForMaskedLM
#from transformers import RobertaModel
#from torch.nn import CrossEntropyLoss
from loss import *
#import wandb
from peft import get_peft_config, get_peft_model,  TaskType, PeftType
from peft import PromptTuningInit, PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig
from PromptTuningClient.PrefixTuning import PrefixTunedRoberta
import openai 
import sys



from scipy.optimize import minimize
import csv
import time


from preprocess_GPT import prepare_and_load_dataset, split_dataset_among_clients, CSV_log, Tracker, task_to_keys, create_batches, CompleteGPT

#from PromptTuningClient_GPT.BBT import ClientBBT
from PromptTuningClient_GPT.BDPL_GPT import ClientBDPL
from PromptTuningClient_GPT.GumbelBDPL_GPT import ClientGumbelBDPL

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument("--task_name", type=str, default=None, help="The name of the glue task.", choices=list(task_to_keys.keys()))
    parser.add_argument("--file_name", type=str, default=None, help="The name of the domain-specific task.")
    parser.add_argument("--low_resource", action="store_true")
    parser.add_argument("--ce_loss", type=bool, default=True)
    parser.add_argument("--sample_size", type=int, default=20, help="IMPORTANT, sample size per batch") # 
    parser.add_argument("--prompt_length", type=int, default=6)
    parser.add_argument("--prompt_learning_rate", type=float, default=5e-5)
    parser.add_argument("--prompt_search_space", type=int, default=20)
    parser.add_argument("--num_train_epochs", type=int, default=30, help="Total number of training epochs to perform.")
    parser.add_argument("--ckpt_path", type=str, default="./ckpts")
    parser.add_argument("--margin", type=float, default=1)
    parser.add_argument("--use_wandb", action="store_false", default=False, help="Whether to run wandb.")
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--max_length", type=int, default=450, help=(
            "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
            " sequences shorter will be padded if `--pad_to_max_lengh` is passed."))
    parser.add_argument("--pad_to_max_length", action="store_true", help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=128, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=32, help="Batch size (per device) for the evaluation dataloader.")
    parser.add_argument("--model_name_or_path", type=str, default='roberta-large', help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay to use.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use.", choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_steps", type=int, default=100, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--k_shot", default=-1, type=int, help="-1 denotes full-shot")
    parser.add_argument("--use_ngram", default=True, type=bool, help="If True, will extract ngrams and use them.")
    parser.add_argument("--api_limit", type=int, default=8000 , help="The limit of the API request")
    # Federated learning
    parser.add_argument("--FL_framework", type=str, default="FedAvg", help="Which Federated Learning Framework: FedAvg, FedSeq")
    parser.add_argument("--num_clients", type=int, default=10 , help="The number of clients in FL.")
    parser.add_argument("--num_activated_clients", type=int, default=10 , help="The number of activated clients in each epoch of FL.")
    parser.add_argument("--num_client_local_step", type=int, default=1000 , help="The number of clients' local update epoch in FL.")
    parser.add_argument("--max_client_train_steps", type=int, default=8000, help="The limit of client's local iteration, per activation")
    parser.add_argument("--local_api_limit", type=int, default=200 , help="The limit of the API request for each client, training epoch/evaluation/testing")
    # prompt tuning method. 
    parser.add_argument("--prompt_tuning_method", type=str, default="BDPL", help="Which white-box tuning method:BBT, BDPL, prefix-tuning, prompt-tuning, " )
    # BDPL
    parser.add_argument("--bdpl_gradient_method", type=str, default="negative", help="negative, zero, normalize" )
    # BBT parameter
    parser.add_argument("--bbt_d", type=int, default=500, help="the d for BBT.")
    parser.add_argument("--bbt_sigma", type=float, default=1.0, help="the sigma for CMAES in BBT.")
    parser.add_argument("--bbt_population_size", type=int, default=20, help="the population size for CMAES in BBT.") #多次采样次数
    # BDPL Gumbel Softmax 
    parser.add_argument("--tau", type=float, default=0.1, help="The temperature of gumbel_softmax")
    # GPT
    #parser.add_argument("--openai_model_name", type=str, default="", help="if not none, will be use chatGPT api. ")
    parser.add_argument("--max_tokens", type=int, default=100, help="max_tokens for the openai api. ")
    parser.add_argument("--top_logprob", type=int, default=10, help="number of the top log_prob  for the openai api. ")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--std", type=float, default=0.01)
    # Early Stop
    parser.add_argument("--early_stop", type=float, default=-1.0, help="stop when the validation result reach")
    # skip training. 
    parser.add_argument("--skip_training", default=False, action="store_true", help="If you don't want to train.") 
    parser.add_argument("--skip_evaluation", default=False, action="store_true", help="If you don't want to evaluate.") 
    parser.add_argument("--skip_test", default=False, action="store_true", help="If you don't want to test.")    
    # trial step 
    parser.add_argument("--trial", action="store_true")
    parser.add_argument("--train_trial_step", type=int, default=10)  # add trial step. 
    parser.add_argument("--eval_trial_step", type=int, default=10)  # add trial step. 
    parser.add_argument("--test_trial_step", type=int, default=10)  # add trial step. 
    # log file. 
    parser.add_argument("--log_file_name", type=str, default="TempResult", help="log file path." )
    args = parser.parse_args()

    args.train_file = './dataset/' + args.file_name + '/train.csv' if args.file_name else None
    args.validation_file = './dataset/' + args.file_name + '/dev.csv' if args.file_name else None
    args.test_file = './dataset/' + args.file_name + '/test.csv' if args.file_name else None

    sanity = not (args.task_name and args.file_name)
    assert sanity

    # Sanity checks
    if args.task_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a task name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."

    # special design for FL. 
    args.k_shot = args.k_shot * args.num_clients  # making each FL hold a k_shot dataset. 

    return args


if __name__ == "__main__":

    args = parse_args()

    # Add GPT
    complete_GPT = CompleteGPT()                                                         # 1.1 add compleGPT. 

    # prepare log
    csv_log = CSV_log(args.log_file_name)
    tracker = Tracker()

    # 0 准备dataset。 
    info1, info2, info3, info4= prepare_and_load_dataset(args)
    accelerator, label_to_id, tokenizer, prompt_length, metric, ngram_list = info1
    hingeloss, ce_loss = info2
    train_dataset, eval_dataset, test_dataset = info3
    train_batches, eval_batches, test_batches, test_batches_mm = info4

    # special variables for record. 
    len_entire_train_dataset = len(train_dataset) 
    best_eval_result = 0
    eval_results = [] # for record. 
    test_results = [] 

    print(len(train_batches["sentence"]))
    print(len(eval_batches["sentence"]))
    print(len(test_batches["sentence"]))
    print(args.trial)
    print(args.train_trial_step)
    print(args.eval_trial_step)
    print(args.test_trial_step)
    #raise Exception()

    # 1 分割 dataset. 按照样本id 平均分配。
    client_trainset_list = split_dataset_among_clients(train_dataset, args.num_clients, mode="random")

    # Ininialize clients
    client_list = []
    for client_idx in range(args.num_clients):
        if args.prompt_tuning_method == "BBT":
            pass
        #    client = ClientBBT(args, accelerator, model, client_trainset_list[client_idx], data_collator, config)
        elif args.prompt_tuning_method == "BDPL":
            client = ClientBDPL(args, accelerator, client_trainset_list[client_idx], ngram_list, complete_GPT)
        elif args.prompt_tuning_method == "GumbelBDPL":
            client = ClientGumbelBDPL(args, accelerator, client_trainset_list[client_idx], ngram_list, complete_GPT)
        client_list.append(client) 

    # 2 写 FL训练的框架。
    if args.prompt_tuning_method == "BDPL":
        average_theta = torch.FloatTensor([[1 / args.prompt_search_space] * args.prompt_search_space] * args.prompt_length)
    if args.prompt_tuning_method == "GumbelBDPL":
        #average_theta = torch.FloatTensor([[1 / args.prompt_search_space] * args.prompt_search_space] * args.prompt_length)
        average_theta = torch.FloatTensor([[2.0] * args.prompt_search_space] * args.prompt_length)
        print(average_theta)
    elif args.prompt_tuning_method == "BBT":
        average_theta = torch.zeros(client_list[0].d)

    # Start the training process. 
    for epoch in range(args.num_train_epochs):
        train_batches = create_batches(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
        train_batches = accelerator.prepare(train_batches)
        
        if args.skip_training: print("skip training...")
        else:
            print(f"start training epoch {epoch}")
            tracker.start_comp_time_tracker()
            if args.FL_framework == "FedAvg":
                # training. 
                client_prompts_probs_list = []
                client_dataset_len_list = []
                for client_idx in random.sample(range(args.num_clients), args.num_activated_clients):
                    # Each client train and update.  
                    client_prompts_probs = client_list[client_idx].local_training(args, None, tokenizer, average_theta, tracker)
                    client_prompts_probs_list.append(client_prompts_probs) #print("client_prompts_probs: \n", client_prompts_probs)
                    # get the weight for averaging. 
                    client_dataset_len_nk = client_list[client_idx].get_len_dataset() 
                    client_dataset_len_list.append(client_dataset_len_nk) #print("weight: \n", weight)

                    # calculate the FL communication 
                    tracker.FL_comm_cost_up += tracker.calculate_comm_size(average_theta)
                    tracker.FL_comm_cost_down += tracker.calculate_comm_size(average_theta)
                    tracker.FL_query_times += 1

                # Fed Average. 
                sampled_client_dataset_len_sum_mt = sum(client_dataset_len_list) 
                average_theta = sum(nk/sampled_client_dataset_len_sum_mt * tensor for nk, tensor in zip(client_dataset_len_list, client_prompts_probs_list)) 

            elif args.FL_framework == "FedSeq":
                for client_idx in range(args.num_clients):
                    average_theta = client_list[client_idx].local_training(args, None, tokenizer, average_theta, tracker) #avg

                    # calculate the FL communication 
                    tracker.FL_comm_cost_up += tracker.calculate_comm_size(average_theta)
                    tracker.FL_comm_cost_down += tracker.calculate_comm_size(average_theta)
                    tracker.FL_query_times += 1

            tracker.stop_comp_time_tracker()
            print(f"End training epoch {epoch}")
        
        if args.skip_evaluation: print("skip evaluation...")
        else:
            print(f" start evaluate. epoch {epoch}")
            # Evaluation. base on differen prompt method selected. 
            if args.prompt_tuning_method == "BBT":
                pass
                #eval_result = ClientBBT.evaluateBBT(args, model, eval_dataloader, metric, ce_loss, config, accelerator, epoch, eval_results, ngram_list, prompts_probs=average_theta, prompt_length=prompt_length, tokenizer=tokenizer)
            elif args.prompt_tuning_method == "BDPL":
                eval_result = client_list[0].evaluateBDPL(args, eval_batches, metric, ce_loss, args, accelerator, epoch, eval_results, ngram_list, prompts_probs=average_theta, prompt_length=prompt_length,tokenizer=tokenizer)
            elif args.prompt_tuning_method == "GumbelBDPL":
                eval_result, eval_prompt_prob  = client_list[0].evaluateGumbelBDPL(args, eval_batches, metric, ce_loss, args, accelerator, epoch, eval_results, ngram_list, prompts_alpha=average_theta, prompt_length=prompt_length,tokenizer=tokenizer)
            else:
                raise Exception("Prompt-tuning method incoorect.")
            
            row =  [epoch, tracker.comp_time,
                    eval_result, 'val_metric_2',
                    tracker.FL_comm_cost_up, tracker.FL_comm_cost_down, tracker.FL_comm_cost(), tracker.FL_query_times, 
                    'LLM_comm_cost_F', "LLM_comm_cost_B", "LLM_comm_cost", complete_GPT.train_api_request.count ]
            csv_log.append_log(row) 
            #print(average_theta)

            if eval_result >= best_eval_result:
                best_eval_result = eval_result
                best_theta = average_theta.clone().detach()
                if args.prompt_tuning_method == "GumbelBDPL":
                    best_prompt_prob = eval_prompt_prob.clone().detach()
                print("best theta")
            if 'cuda' in str(args.device):
                torch.cuda.empty_cache()
            if complete_GPT.train_api_request.count >= args.api_limit:
                break
            print(average_theta[0])

            # early stop. 
            if args.early_stop > 0:
                if eval_result > args.early_stop:
                    break
            print("End evaluate. ")

    print(f"the skip test is {args.skip_test}")
    if args.skip_test: print("skip test...")
    else:
        print("start test. ")
        if args.prompt_tuning_method == "BBT":
            pass
        #    test_result = ClientBBT.testBBT(args, model, test_dataloader, metric, accelerator, epoch, test_results, ngram_list, prompts_probs=best_theta, prompt_length=prompt_length, tokenizer=tokenizer, test_dataloader_mm=test_dataloader_mm)
        elif args.prompt_tuning_method == "BDPL":
            test_result = client_list[0].testBDPL(args, test_batches, metric, accelerator, epoch, test_results, prompts_probs=best_theta, prompt_length=prompt_length, tokenizer=tokenizer, linear_layer=None, prompts=None, label_to_id=label_to_id, test_batches_mm=test_batches_mm)
        elif args.prompt_tuning_method == "GumbelBDPL":
            test_result = client_list[0].testGumbelBDPL(args, test_batches, metric, accelerator, epoch, test_results, prompts_probs=best_prompt_prob, prompt_length=prompt_length, tokenizer=tokenizer, linear_layer=None, prompts=None, label_to_id=label_to_id, test_batches_mm=test_batches_mm)   # prompts_alpha
        else:
            raise Exception("Prompt-tuning method incoorect.")
        
        # add the log for the final.  
        row =  [-100, tracker.comp_time,
                    test_result, test_results,
                    tracker.FL_comm_cost_up, tracker.FL_comm_cost_down, tracker.FL_comm_cost(), tracker.FL_query_times, 
                    'LLM_comm_cost_F', "LLM_comm_cost_B", "LLM_comm_cost", complete_GPT.train_api_request.count ]
        csv_log.append_log(row)
        print(best_theta)
        print(torch.sum(best_theta))
