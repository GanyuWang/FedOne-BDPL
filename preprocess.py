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

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

DOMAIN_DATASET = ['CI', 'SE', 'RCT', 'HP']

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

LABEL2ID_CONFIG = {
    "mnli": {" no": 0, " maybe": 1, " yes": 2},
    "qqp": {" no": 0, " yes": 1},
    "sst2": {" terrible": 0, " great": 1},
    "mrpc": {" no": 0, " yes": 1},
    "cola": {" no": 0, " yes": 1},
    "wnli": {" no": 0, " yes": 1},
    "qnli": {" yes": 0, " no": 1},
    "rte": {" yes": 0, " no": 1},
    "CI": {' background': 0, ' comparison': 1, ' extension': 2, ' future': 3, ' motivation': 4, ' use': 5},
    "SE": {' comparison': 0, ' conjunction': 1, ' evaluation': 2, ' feature': 3, ' hyponym': 4, ' part': 5, ' function': 6},
    "RCT": {' background': 0, ' conclusion': 1, ' method': 2, ' objective': 3, ' result': 4} ,
    "HP": {' unhelpful': 0, ' helpful': 1}, # review helpfulness
    "imdb": {" terrible": 0, " great": 1},
    "cr": {" terrible": 0, " great": 1},
    "mr": {" terrible": 0, " great": 1},
    "mpqa": {" terrible": 0, " great": 1}
}

LABEL_CONVERT = {
    "mnli": {0: ' no', 1: ' maybe', 2: ' yes'},
    "qqp": {0: ' no', 1: ' yes'},
    "sst2": {0: ' terrible', 1: ' great'},
    'mrpc': {0: ' no', 1: ' yes'},
    'cola': {0: ' no', 1: ' yes'},
    'wnli': {0:  ' no', 1: ' yes'},
    'qnli': {0: ' yes', 1: ' no'},
    'rte': {0: ' yes', 1: ' no'},
    'CI': {'Background': ' background', 'CompareOrContrast': ' comparison', 'Extends': ' extension', 'Future': ' future', 'Motivation': ' motivation', 'Uses': ' use'},
    'SE': {'COMPARE': ' comparison', 'CONJUNCTION': ' conjunction', 'EVALUATE-FOR': ' evaluation', 'FEATURE-OF': ' feature', 'HYPONYM-OF': ' hyponym', 'PART-OF': ' part', 'USED-FOR': ' function'},
    'RCT': {'BACKGROUND': ' background', 'CONCLUSIONS': ' conclusion', 'METHODS': ' method', 'OBJECTIVE': ' objective', 'RESULTS': ' result'},
    'HP': {False: ' unhelpful', True: ' helpful'},
}

TEMPLATE_CONFIG = {
    "mnli": " entailment? [MASK].",
    "qqp": "? [MASK],",
    "sst2": " It was [MASK].",
    "mrpc": "? [MASK],",
    "cola": " correct? [MASK].",
    "wnli": " entailment? [MASK].",
    "qnli": " entailment? [MASK].",
    "rte": " entailment? [MASK].",
    "CI": " What is the intent? [MASK].", 
    "SE": " What is the relation? [MASK].",
    "RCT": " It is [MASK]. ",
    "HP": " It is [MASK].",
    "imdb": "It was [MASK].",
    "cr": "It was [MASK].",
}

def solve_v_total_exact(prompt_emb):
    k = 1
    a, b = 0, 0

    b = prompt_emb.max()
    def f(v):
        s = (prompt_emb - v).clamp(0, 1).sum()
        return s - k
    itr = 0

    v = 0
    while (1):
        itr += 1
        v = (a + b) / 2
        obj = f(v)
        if abs(obj) < 1e-3 or itr > 20:
            break
        if obj < 0:
            b = v
        else:
            a = v
    return v, itr

def constrainScoreByWholeExact(prompt_embeds):
    for i in range(len(prompt_embeds)):
        v, itr = solve_v_total_exact(prompt_embeds[i])
        prompt_embeds[i].sub_(v).clamp_(0, 1)
    

def pmi(args):

    result=[]
    if args.file_name:
        with open("./pmi/" + args.file_name.lower() + ".txt",'r') as f:
            for line in f:
                result = result + (list(line.strip('\n').split(',')))
    elif args.task_name:
        with open("./pmi/" + args.task_name.lower() + ".txt",'r') as f:
            for line in f:
                result = result + (list(line.strip('\n').split(',')))

    unique = []
    [unique.append(i) for i in result if not i in unique]
    ngram_index_list = list(map(int, unique))
    return ngram_index_list

def counter(func):
    def wrapper(*args, **kwargs):
        wrapper.count = wrapper.count + 1
        res = func(*args, **kwargs)
        if wrapper.count % 100 == 0:
            print ("{0} has been used: {1}x".format(func.__name__, wrapper.count))
        return res
    wrapper.count = 0
    return wrapper

class ApiCallLimitError(Exception):
    pass



def prepare_and_load_dataset(args):
    
    ngram_list = pmi(args)

    assert args.task_name != 'stsb'
    ce_loss_string = 'True' if args.ce_loss else 'False'

    # specify a unique experiment_id for load_metric() otherwise will cause ERROR when having multiple run on a same server!
    task_name = args.task_name if args.task_name else args.train_file
    args.unique_task_name = task_name.replace("/", ".")
    args.experiment_id = task_name + str(args.prompt_length) + str(args.prompt_learning_rate) \
                         + str(args.num_train_epochs) + str(args.seed) + str(args.prompt_search_space) + ce_loss_string #'dataset/CI/train.csv1020.0013042160.01falseFALSE'

    if args.use_wandb:
        args.group_name = "RoBERTa_BDPL_" + task_name
        wandb.init(config=args, project="blackbox_prompt", group=args.group_name)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    accelerator = Accelerator()
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # download the dataset.
    if args.task_name is not None:
        if args.task_name in task_to_keys.keys():
            raw_datasets = load_dataset("glue", args.task_name)
        else:
            raise(NotImplementedError)
    else:
        # Loading the dataset from local csv or json file.
        data_files = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        if args.test_file is not None:
            data_files["test"] = args.test_file
        extension = (args.train_file if args.train_file is not None else args.valid_file).split(".")[-1]
        raw_datasets = load_dataset(extension, data_files=data_files)

    # Some models have set the order of the labels to use, so let's make sure we do use it.
    label_to_id = None
    if args.task_name:
        label_to_id = LABEL2ID_CONFIG[args.task_name]
    elif args.file_name:
        label_to_id = LABEL2ID_CONFIG[args.file_name]
    num_labels = len(label_to_id)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    config = RobertaConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)

    # init model
    model = RobertaForMaskedLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
    )
    
    args.device = torch.device("cuda", args.cuda)
    model.to(args.device)

    if label_to_id is not None:
        model.config.label2id = label_to_id
        model.config.id2label = {id: label for label, id in config.label2id.items()}

    
    prompt_length = args.prompt_length
    hingeloss = MarginLoss(margin=args.margin, target=False)
    ce_loss = CrossEntropyLoss()

    # Preprocessing the datasets
    if args.task_name is not None:
        sentence1_key, sentence2_key = task_to_keys[args.task_name]
    else:
        non_label_column_names = [name for name in raw_datasets["train"].column_names if name != "label"]
        if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
            sentence1_key, sentence2_key = "sentence1", "sentence2"
        else:
            if len(non_label_column_names) >= 2:
                sentence1_key, sentence2_key = non_label_column_names[:2]
            else:
                sentence1_key, sentence2_key = non_label_column_names[0], None

    padding = "max_length" if args.pad_to_max_length else False

    
    def preprocess_function(examples):
        # Tokenize the texts
        if args.low_resource:
            train_random_samples = random.sample(range(0, len(examples["label"])), len(examples["label"])//10)
            for key in examples.keys():
                examples[key] = [examples[key][k] for k in train_random_samples]

        if args.file_name == 'HP':
            for k in range(len(examples["text_a"])):
                if examples["text_a"][k] == None:
                    examples["text_a"].remove(examples["text_a"][k])
                    examples["label"].remove(examples["label"][k])
                    break

        if args.task_name is not None:
            template_cfg = TEMPLATE_CONFIG[args.task_name]
        elif args.file_name is not None:
            template_cfg = TEMPLATE_CONFIG[args.file_name]
        template_base = template_cfg.replace('[MASK]', tokenizer.mask_token)

        if sentence2_key:
            sent1_list = []
            for sent1 in examples[sentence1_key]:
                sent1_list.append(sent1 + template_base)
            texts = (sent1_list, examples[sentence2_key])
        else:
            template = [template_base] * len(examples[sentence1_key])
            texts = (examples[sentence1_key], template)
        result = tokenizer(*texts, padding=padding, max_length=args.max_length, truncation=True, add_special_tokens=False)

        texts = []
        template = [template_base] * len(examples[sentence1_key])
        if sentence2_key:
            for tuple_ in list(zip(examples[sentence1_key], template, examples[sentence2_key])):
                sent_1 = tokenizer.tokenize(tuple_[0])[:200]
                new_sent_1 = tokenizer.convert_tokens_to_string(sent_1)
                sent_2 = tokenizer.tokenize(tuple_[2])[:200]
                new_sent_2 = tokenizer.convert_tokens_to_string(sent_2)
                texts.append(new_sent_1 + tokenizer.sep_token + new_sent_2 + tuple_[1])
        else:
            for tuple_ in list(zip(examples[sentence1_key], template)):
                sent_1 = tokenizer.tokenize(tuple_[0])[:400]
                new_sent_1 = tokenizer.convert_tokens_to_string(sent_1)
                texts.append(new_sent_1 + tuple_[1])
        result = tokenizer(texts, padding=padding, max_length=args.max_length, truncation=True)

        if args.task_name:
            label_list = []
            for raw_label in examples["label"]:
                label = LABEL_CONVERT[args.task_name][raw_label]
                target_encodings = tokenizer.encode(str(label).lower(), add_special_tokens=False)
                label_list.append(target_encodings[0])
            result["labels"] = torch.tensor(label_list)

        elif args.file_name in DOMAIN_DATASET:
            label_list = []
            for raw_label in examples["label"]:
                label = LABEL_CONVERT[args.file_name][raw_label]
                target_encodings = tokenizer.encode(str(label).lower(), add_special_tokens=False)
                label_list.append(target_encodings[0])
            result["labels"] = torch.tensor(label_list)
        else:
            target_encodings = tokenizer.batch_encode_plus(examples["label"], add_special_tokens=False)
            result["labels"]= torch.tensor(target_encodings['input_ids']).squeeze(dim=1).to(args.device)
            
        return result
    
    # 
    def preprocess_function_k_shot(examples):
        random_indices = list(range(0, len(examples["label"])))
        random.shuffle(random_indices)

        new_examples = {}
        for key in examples.keys():
            new_examples[key] = []
        label_count = {}

        for index in random_indices:
            label = examples['label'][index]
            if label not in label_count:
                label_count[label] = 0

            if label_count[label] < args.k_shot:
                for key in examples.keys():
                    new_examples[key].append(examples[key][index])
                label_count[label] += 1
        
        print("Finish few-shot sampling!")

        result = preprocess_function(new_examples)
        return result

    with accelerator.main_process_first():
        if args.k_shot >= 0:
            # k-shot learning
            raw_train_dataset_split = raw_datasets["train"].train_test_split(test_size=0.5)
            raw_train_dataset = raw_train_dataset_split['train']
            raw_eval_dataset = raw_train_dataset_split['test']
            train_dataset = raw_train_dataset.map(
                preprocess_function_k_shot,
                batched=True,
                batch_size=100000,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
            eval_dataset = raw_eval_dataset.map(
                preprocess_function_k_shot,
                batched=True,
                batch_size=100000,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
            if args.task_name == 'mnli':
                test_dataset = raw_datasets["validation_matched"].map(
                    preprocess_function,
                    batched=True,
                    remove_columns=raw_datasets["train"].column_names,
                    load_from_cache_file=False,
                    desc="Running tokenizer on dataset",
                )
                test_dataset_mm = raw_datasets["validation_mismatched"].map(
                    preprocess_function,
                    batched=True,
                    remove_columns=raw_datasets["train"].column_names,
                    load_from_cache_file=False,
                    desc="Running tokenizer on dataset",
                )
            else:
                test_dataset = raw_datasets["validation"].map(
                    preprocess_function,
                    batched=True,
                    remove_columns=raw_datasets["train"].column_names,
                    load_from_cache_file=False,
                    desc="Running tokenizer on dataset",
                )
        else:
            train_dataset = raw_datasets["train"].map(
                preprocess_function,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
            eval_dataset = raw_datasets["validation"].map(
                preprocess_function,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
            test_dataset = raw_datasets["test" if args.file_name != None else "validation"].map(
                preprocess_function,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=False,
                desc="Running tokenizer on dataset",
            )
        print("length of train data",len(train_dataset))
        print("length of eval data",len(eval_dataset))
        print("length of test data",len(test_dataset))

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # DataLoaders creation:
    if args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))


    # split dataset. 
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    if args.task_name == 'mnli':
        test_dataloader_mm = DataLoader(test_dataset_mm, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
        test_dataloader_mm = accelerator.prepare(test_dataloader_mm)
    else:
        test_dataloader_mm = None
    model, train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(model, train_dataloader, eval_dataloader, test_dataloader)

    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name, experiment_id=args.experiment_id)
    elif args.file_name in DOMAIN_DATASET:
        metric = load_metric('f1', args.experiment_id)
    else:
        metric = load_metric('accuracy', args.experiment_id)

    return (accelerator, label_to_id, tokenizer, config, model, prompt_length, metric, ngram_list), \
           (hingeloss, ce_loss), \
           (train_dataset, eval_dataset, test_dataset, data_collator), \
           (train_dataloader, eval_dataloader, test_dataloader, test_dataloader_mm) \


@counter
def train_api_request(model, input_ids=None, attention_mask=None):
    sequence_output = model(input_ids=input_ids, attention_mask=attention_mask)
    return sequence_output


# Split the dataset. 
def split_dataset_among_clients(dataset, num_clients, mode="seq"):

    assert len(dataset) > num_clients

    # Determine the indices for splitting
    indices = list(range(len(dataset)))
    if mode == "random":
        random.shuffle(indices)  # Shuffle only in random mode

    # Calculate the size of each subset
    subset_size = len(dataset) // num_clients
    extra_samples = len(dataset) % num_clients

    # Split the dataset into subsets
    subsets = []
    start_idx = 0
    for i in range(num_clients):
        if i < extra_samples:
            end_idx = start_idx + subset_size + 1
        else:
            end_idx = start_idx + subset_size

        # Select a range of indices from the dataset to create subsets
        subset_indices = indices[start_idx:end_idx]
        subset = dataset.select(subset_indices)
        subsets.append(subset)
        start_idx = end_idx

    return subsets

def print_trainable_parameters(model):
    print("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel()}")




class CSV_log: 
    def __init__(self, log_file_name):
        self.log_file_name = log_file_name
        self.field = ["epoch", "comp_time",
                      'val_metric_1', 'val_metric_2',
                      'FL_comm_cost_up', "FL_comm_cost_down", "FL_comm_cost", 'FL_query_times', 
                      'LLM_comm_cost_up', "LLM_comm_cost_down", "LLM_comm_cost", 'LLM_query_times' ]
        for e in self.field:
            print(e, end=", ")
        print()
        if log_file_name == "TempResult":
            with open(f"{log_file_name}.csv", 'w') as f:
                write = csv.writer(f)
                write.writerow(self.field)
        else: 
            with open(f"{log_file_name}.csv", 'x') as f:
                write = csv.writer(f)
                write.writerow(self.field)

    def append_log(self, row):
        with open(f"{self.log_file_name}.csv", 'a') as f:
            write = csv.writer(f)
            write.writerow(row)
        for e in row:
            print(e, end=", ")
        print()

class Tracker:
    def __init__(self):
        self.comp_time = 0
        self.FL_comm_cost_up = 0
        self.FL_comm_cost_down = 0
        self.FL_query_times = 0
        self.LLM_comm_cost_up = 0
        self.LLM_comm_cost_down = 0
        self.LLM_query_times = 0
        
    def start_comp_time_tracker(self):
        self.start_time = time.time()
    def stop_comp_time_tracker(self):
        t = time.time() - self.start_time
        self.comp_time += t
        del self.start_time

    def calculate_comm_size(self, x):
        return x.element_size() * x.nelement() / 1048576

    def FL_comm_cost(self):
        return self.FL_comm_cost_up + self.FL_comm_cost_down
    def LLM_comm_cost(self): 
        return self.LLM_comm_cost_up + self.LLM_comm_cost_down 


