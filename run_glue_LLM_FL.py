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
from PromptTuningClient.PrefixTuning import PrefixTunedRoberta




from scipy.optimize import minimize
import csv
import time


from preprocess import prepare_and_load_dataset, train_api_request, split_dataset_among_clients, CSV_log, Tracker, task_to_keys


from PromptTuningClient.BBT import ClientBBT
from PromptTuningClient.BDPL import ClientBDPL, evaluateBDPL, testBDPL
from PromptTuningClient.Gumbel_BDPL import ClientGumbelBDPL, evaluateGumbelBDPL, testGumbelBDPL
from PromptTuningClient.PrefixTuning import ClientPrefixTuning, evaluatePrefixTuning, testPrefixTuning
from PromptTuningClient.PromptTuning import ClientPromptTuning, evaluatePromptTuning, testPromptTuning




def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument("--task_name", type=str, default=None, help="The name of the glue task.", choices=list(task_to_keys.keys()))
    parser.add_argument("--file_name", type=str, default=None, help="The name of the domain-specific task.")
    parser.add_argument("--low_resource", action="store_true")
    parser.add_argument("--ce_loss", type=bool, default=True)
    parser.add_argument("--sample_size", type=int, default=20, help="IMPORTANT, sample size per batch") # #多次采样次数
    parser.add_argument("--prompt_length", type=int, default=6)
    parser.add_argument("--prompt_learning_rate", type=float, default=5e-5)
    parser.add_argument("--prompt_search_space", type=int, default=20)
    parser.add_argument("--num_train_epochs", type=int, default=30, help="Total number of training epochs to perform.")
    parser.add_argument("--ckpt_path", type=str, default="./ckpts")
    parser.add_argument("--margin", type=float, default=1)
    parser.add_argument("--trial", action="store_true")
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
    # Early Stop
    parser.add_argument("--early_stop", type=float, default=-1.0, help="stop when the validation result reach") # 
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

    # prepare log
    csv_log = CSV_log(args.log_file_name)
    tracker = Tracker()

    # 0 准备dataset。 
    info1, info2, info3, info4 = prepare_and_load_dataset(args)
    accelerator, label_to_id, tokenizer, config, model, prompt_length, metric, ngram_list = info1
    hingeloss, ce_loss = info2
    train_dataset, eval_dataset, test_dataset, data_collator = info3
    train_dataloader, eval_dataloader, test_dataloader, test_dataloader_mm = info4

    # special variables for record. 
    len_entire_train_dataset = len(train_dataset) 
    best_eval_result = 0
    eval_results = [] # for record. 
    test_results = []

    # Black-box tuning. 
    if args.prompt_tuning_method in ["BDPL", "GumbelBDPL", "BBT"]:
        model.eval()
        for name, param in model.named_parameters():
            param.requires_grad = False

    # White-box  Prompt tuning. 
    elif args.prompt_tuning_method == "prompt-tuning":
        # Prompt Tuning. Configuration. 
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=args.prompt_length,
            prompt_tuning_init_text="Classify if the tweet is a complaint or not:", # this should be updated! later.
            tokenizer_name_or_path=args.model_name_or_path,
        )
        # local trainable = average theta. 
        model = get_peft_model(model, peft_config)
    elif args.prompt_tuning_method == "prefix-tuning":
        model = PrefixTunedRoberta(args, model, config, args.prompt_length).to(args.device)
    print(f"The prompt tuning method is: {args.prompt_tuning_method}")

    # 1 分割 dataset. 按照样本id 平均分配。
    client_trainset_list = split_dataset_among_clients(train_dataset, args.num_clients, mode="random")

    # Ininialize clients
    client_list = []
    for client_idx in range(args.num_clients):
        if args.prompt_tuning_method == "BBT":
            client = ClientBBT(args, accelerator, model, client_trainset_list[client_idx], data_collator, config)
        elif args.prompt_tuning_method == "BDPL":
            client = ClientBDPL(args, accelerator, client_trainset_list[client_idx], data_collator, config, ngram_list)
        elif args.prompt_tuning_method == "GumbelBDPL":
            client = ClientGumbelBDPL(args, accelerator, client_trainset_list[client_idx], data_collator, config, ngram_list)
        elif args.prompt_tuning_method == "prefix-tuning":
            client = ClientPrefixTuning(args, accelerator, model, client_trainset_list[client_idx], data_collator, config)
        elif args.prompt_tuning_method == "prompt-tuning":
            client = ClientPromptTuning(args, accelerator, model, client_trainset_list[client_idx], data_collator, config)
        client_list.append(client) 

    # 2 写 FL训练的框架。
    if args.prompt_tuning_method == "BDPL":
        average_theta = torch.FloatTensor([[1 / args.prompt_search_space] * args.prompt_search_space] * args.prompt_length)
    if args.prompt_tuning_method == "GumbelBDPL":
        average_theta = torch.FloatTensor([[1 / args.prompt_search_space] * args.prompt_search_space] * args.prompt_length)*0.01
    elif args.prompt_tuning_method == "BBT":
        average_theta = torch.zeros(client_list[0].d)
    elif args.prompt_tuning_method == "prompt-tuning":
        average_theta = model.prompt_encoder.default.embedding.weight.clone().detach()
    elif args.prompt_tuning_method == "prefix-tuning":
        average_theta = model.trainable_params.clone().detach()
    
    # Start the training process. 
    for epoch in range(args.num_train_epochs):
        tracker.start_comp_time_tracker()
        if args.FL_framework == "FedAvg":
            # training. 
            client_prompts_probs_list = []
            client_dataset_len_list = []
            for client_idx in random.sample(range(args.num_clients), args.num_activated_clients):
                # Each client train and update.  
                client_prompts_probs = client_list[client_idx].local_training(args, model, tokenizer, average_theta, tracker)
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
            if args.prompt_tuning_method == "prompt-tuning":
                model.prompt_encoder.default.embedding.weight.data = average_theta
            elif args.prompt_tuning_method == "prefix-tuning":
                model.trainable_params.data = average_theta 

        elif args.FL_framework == "FedSeq":
            for client_idx in range(args.num_clients):
                average_theta = client_list[client_idx].local_training(args, model, tokenizer, average_theta, tracker) #avg
                if args.prompt_tuning_method == "prompt-tuning":
                    model.prompt_encoder.default.embedding.weight.data = average_theta
                elif args.prompt_tuning_method == "prefix-tuning":
                    model.trainable_params.data = average_theta

                # calculate the FL communication 
                tracker.FL_comm_cost_up += tracker.calculate_comm_size(average_theta)
                tracker.FL_comm_cost_down += tracker.calculate_comm_size(average_theta)
                tracker.FL_query_times += 1


        tracker.stop_comp_time_tracker()

        # Evaluation. base on differen prompt method selected. 
        if args.prompt_tuning_method == "BBT":
            eval_result = ClientBBT.evaluateBBT(args, model, eval_dataloader, metric, ce_loss, config, accelerator, epoch, eval_results, ngram_list, prompts_probs=average_theta, prompt_length=prompt_length, tokenizer=tokenizer)
        elif args.prompt_tuning_method == "BDPL":
            eval_result = evaluateBDPL(args, model, eval_dataloader, metric, ce_loss, config, accelerator, epoch, eval_results, ngram_list, prompts_probs=average_theta, prompt_length=prompt_length, tokenizer=tokenizer)
        elif args.prompt_tuning_method == "GumbelBDPL":
            eval_result = evaluateGumbelBDPL(args, model, eval_dataloader, metric, ce_loss, config, accelerator, epoch, eval_results, ngram_list, prompts_alpha=average_theta, prompt_length=prompt_length, tokenizer=tokenizer)   # prompts_alpha
        elif args.prompt_tuning_method == "prefix-tuning":
            eval_result = evaluatePrefixTuning(args, model, eval_dataloader, metric, ce_loss, config, accelerator, epoch, eval_results, prompts_probs=average_theta, tokenizer=tokenizer)
        elif args.prompt_tuning_method == "prompt-tuning":
            eval_result = evaluatePromptTuning(args, model, eval_dataloader, metric, ce_loss, config, accelerator, epoch, eval_results, prompts_probs=average_theta, prompt_length=prompt_length, tokenizer=tokenizer)
        else:
            raise Exception("Prompt-tuning method incoorect.")
       
        row =  [epoch, tracker.comp_time,
                eval_result, 'val_metric_2',
                tracker.FL_comm_cost_up, tracker.FL_comm_cost_down, tracker.FL_comm_cost(), tracker.FL_query_times, 
                'LLM_comm_cost_F', "LLM_comm_cost_B", "LLM_comm_cost", train_api_request.count ]
        csv_log.append_log(row) 

        #print(average_theta)

        if eval_result >= best_eval_result:
            best_eval_result = eval_result
            best_theta = average_theta.clone().detach()
            print("best theta")
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        if train_api_request.count >= args.api_limit:
            break
        print(average_theta[0])

        # early stop. 
        if args.early_stop > 0:
            if eval_result > args.early_stop:
                break

    if args.prompt_tuning_method == "prompt-tuning":
        model.prompt_encoder.default.embedding.weight.data.copy_(best_theta.data)

    if args.prompt_tuning_method == "prefix-tuning":
        model.trainable_params.copy_from(best_theta)

    if args.prompt_tuning_method == "BBT":
        test_result = ClientBBT.testBBT(args, model, test_dataloader, metric, accelerator, epoch, test_results, ngram_list, prompts_probs=best_theta, prompt_length=prompt_length, tokenizer=tokenizer, test_dataloader_mm=test_dataloader_mm)
    elif args.prompt_tuning_method == "BDPL":
        test_result = testBDPL(args, model, test_dataloader, metric, accelerator, epoch, test_results, ngram_list, prompts_probs=best_theta, prompt_length=prompt_length, tokenizer=tokenizer, test_dataloader_mm=test_dataloader_mm)
    elif args.prompt_tuning_method == "GumbelBDPL":
        test_result = testGumbelBDPL(args, model, test_dataloader, metric, accelerator, epoch, test_results, ngram_list, prompts_alpha=best_theta, prompt_length=prompt_length, tokenizer=tokenizer, test_dataloader_mm=test_dataloader_mm)   # prompts_alpha
    elif args.prompt_tuning_method == "prefix-tuning":
        test_result = testPrefixTuning(args, model, test_dataloader, metric, accelerator, epoch, test_results, ngram_list, prompts_probs=best_theta, prompt_length=prompt_length, tokenizer=tokenizer, test_dataloader_mm=test_dataloader_mm)
    elif args.prompt_tuning_method == "prompt-tuning":
        test_result = testPromptTuning(args, model, test_dataloader, metric, accelerator, epoch, test_results, prompts_probs=best_theta, prompt_length=prompt_length, tokenizer=tokenizer, test_dataloader_mm=test_dataloader_mm)
    else:
        raise Exception("Prompt-tuning method incoorect.")
    
    # add the log for the final. 
    row =  [-100, tracker.comp_time,
                test_result, test_results,
                tracker.FL_comm_cost_up, tracker.FL_comm_cost_down, tracker.FL_comm_cost(), tracker.FL_query_times, 
                'LLM_comm_cost_F', "LLM_comm_cost_B", "LLM_comm_cost", train_api_request.count ]
    csv_log.append_log(row)
    print(best_theta)