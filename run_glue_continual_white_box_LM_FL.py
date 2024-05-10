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
    
def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a text classification task")
    parser.add_argument("--task_name", type=str, default=None, help="The name of the glue task.", choices=list(task_to_keys.keys()))
    parser.add_argument("--file_name", type=str, default=None, help="The name of the domain-specific task.")
    parser.add_argument("--low_resource", action="store_true")
    parser.add_argument("--ce_loss", type=bool, default=True)
    parser.add_argument("--sample_size", type=int, default=20, help="IMPORTANT, sample size per batch")
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
    parser.add_argument("--num_client_local_step", type=int, default=1000 , help="The number of clients' local update steps in FL.")
    parser.add_argument("--max_client_train_steps", type=int, default=8000, help="Total number of training steps to perform. If provided, overrides num_train_epochs.")
    # white box prompt tuning. 
    parser.add_argument("--prompt_tuning_method", type=str, default="BDPL", help="Which white-box tuning method:BBT, BDPL, prefix-tuning, prompt-tuning, " )
    
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

def pmi():
    args = parse_args()
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

ngram_list = pmi()


def prepare_and_load_dataset(args):
    
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

    return (accelerator, label_to_id, tokenizer, config, model, prompt_length, metric), \
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

class ClientBDPL:
    def __init__(self, args, accelerator, client_partial_train_dataset, data_collator):
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
    
    def get_len_dataset(self):
        return len(self.dataset)

    def local_training(self, args, model, tokenizer, average_theta):

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
                                    prompts_discrete_indices_ngram_list.append(ngram_list[idx]) 
                                prompts_discrete_ngram_indices = torch.tensor(prompts_discrete_indices_ngram_list)
                                cur_input_ids = torch.cat([torch.zeros(bsz, 1, dtype=torch.long).to(args.device), prompts_discrete_ngram_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1)
                            else: 
                                cur_input_ids = torch.cat([torch.zeros(bsz, 1, dtype=torch.long).to(args.device), prompts_discrete_indices.unsqueeze(0).repeat(bsz, 1).to(args.device), batch['input_ids'][:, 1:]], dim=1) # CLS + Discrete Prompt + input_ids

                            cur_attention_mask = torch.cat([torch.ones(bsz, 1).to(args.device), torch.ones(bsz, prompt_length).to(args.device), batch["attention_mask"][:, 1:]],dim=1) # [0, 1(prompt length), original_attention_mask]
                            mask_pos = np.where(np.array(cur_input_ids.cpu()) == tokenizer.mask_token_id)     # 找到 mask position. 
                            mask_pos = torch.tensor(mask_pos[-1]) 
                            sequence_output = train_api_request(model, input_ids=cur_input_ids, attention_mask=cur_attention_mask)
                            last_hidden_state = sequence_output[0].squeeze()
                            logits = last_hidden_state[torch.arange(last_hidden_state.size(0)), mask_pos]

                            label_keys = list(label_to_id.keys())
                            label_map = {}
                            for target in label_keys:
                                label_map[tokenizer.encode(target, add_special_tokens=False)[0]] = label_to_id[target]
                            
                            converted_target = label.clone()
                            for key, val in label_map.items():
                                converted_target[label == key] = val
                            interest_index = list(label_map.keys())
                            logits = logits[:, interest_index]
                            pred = logits.argmax(dim=-1)

                            if args.ce_loss:
                                loss = ce_loss(logits.view(-1, config.num_labels), converted_target)
                            else:
                                loss = hingeloss(logits, converted_target)
                            loss_list.append(loss.item())

                            if train_api_request.count >= args.api_limit:
                                raise ApiCallLimitError()

                        loss_avg = sum(loss_list) / args.sample_size
                        
                        self.prompt_optimizer.zero_grad()

                        derivative = (-1 / self.prompts_probs).repeat(args.sample_size, 1, 1)
                        for k, prompts_discrete_indices in enumerate(prompts_discrete_indices_list):
                            for i in range(prompt_length):
                                derivative[k][i][prompts_discrete_indices[i]] *= -1

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


class ClientPromptTuning:
    def __init__(self, args, accelerator, model, client_partial_train_dataset, data_collator):

        # optimizer. 
        prompt_parameters = [param for param in model.parameters() if param.requires_grad]
        self.prompt_optimizer = torch.optim.AdamW(prompt_parameters, lr=args.prompt_learning_rate)
        # FL parameter. 
        self.num_local_step = args.num_client_local_step
        self.dataset = client_partial_train_dataset  # Local dataset for the client
        self.train_dataloader = DataLoader(self.dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size)
        self.train_dataloader = accelerator.prepare(self.train_dataloader)
        self.completed_steps = 0
    
    def get_len_dataset(self):
        return len(self.dataset)

    def local_training(self, args, model, tokenizer, average_theta):

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
                    label_keys = list(label_to_id.keys())
                    label_map = {}
                    for target in label_keys:
                        label_map[tokenizer.encode(target, add_special_tokens=False)[0]] = label_to_id[target]

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
                        loss = ce_loss(logits.view(-1, config.num_labels), converted_target)
                    else:
                        loss = hingeloss(logits, converted_target)

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


class ClientPrefixTuning:
    def __init__(self, args, accelerator, model, client_partial_train_dataset, data_collator):
        pass
    class PrefixTunedRoberta(RobertaModel):
        def __init__(self, config):
            super().__init__(config)
            self.prefix_length = 10  # Length of the prefix
            self.prefix_embeddings = torch.nn.Embedding(self.prefix_length, config.hidden_size)
            self.prefix_embeddings.weight.requires_grad = True  # Make prefix embeddings trainable

        def forward(self, input_ids, **kwargs):
            # Generate prefix embeddings
            batch_size = input_ids.shape[0]
            prefix_ids = torch.arange(self.prefix_length).unsqueeze(0).repeat(batch_size, 1).to(input_ids.device)
            prefix_embeddings = self.prefix_embeddings(prefix_ids)

            # Process the input embeddings
            inputs_embeds = self.embeddings(input_ids)
            extended_inputs_embeds = torch.cat([prefix_embeddings, inputs_embeds], dim=1)

            # Continue with the forward pass as usual
            outputs = super().forward(inputs_embeds=extended_inputs_embeds, **kwargs)
            return outputs

def evaluate(args,  model, eval_dataloader, metric, ce_loss,config, accelerator, epoch, results, prompts_probs=None, prompt_length=None,tokenizer=None):
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

def test(args, model, test_dataloader, metric, accelerator, epoch, results, prompts_probs=None, prompt_length=None, tokenizer=None, test_dataloader_mm=None):
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

if __name__ == "__main__":

    args = parse_args()

    # 0 准备dataset。 
    info1, info2, info3, info4 = prepare_and_load_dataset(args)
    accelerator, label_to_id, tokenizer, config, model, prompt_length, metric = info1
    hingeloss, ce_loss = info2
    train_dataset, eval_dataset, test_dataset, data_collator = info3
    train_dataloader, eval_dataloader, test_dataloader, test_dataloader_mm = info4

    # special variables for record. 
    len_entire_train_dataset = len(train_dataset) 
    best_eval_result = 0
    eval_results = [] # for record. 
    test_results = []

    

    # Black-box tuning. 
    if args.prompt_tuning_method in ["BDPL", "BBT"]:
        model.eval()
        for name, param in model.named_parameters():
            param.requires_grad = False

    # White-box  Prompt tuning. 
    if args.prompt_tuning_method == "prompt-tuning":
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
        pass
    
    # 1 分割 dataset. 按照样本id 平均分配。
    client_trainset_list = split_dataset_among_clients(train_dataset, args.num_clients, mode="random")

    # Ininialize clients
    client_list = []
    for client_idx in range(args.num_clients):
        if args.prompt_tuning_method == "BDPL":
            client = ClientBDPL(args, accelerator, client_trainset_list[client_idx], data_collator)
        elif args.prompt_tuning_method == "prompt-tuning":
            client = ClientPromptTuning(args, accelerator, model, client_trainset_list[client_idx], data_collator)
        elif args.prompt_tuning_method == "prefix-tuning":
            client = ClientPrefixTuning(args, accelerator, model, client_trainset_list[client_idx], data_collator)
        client_list.append(client) 
    print(f"The prompt tuning method is: {args.prompt_tuning_method}")

    # 2 写 FL训练的框架。
    if args.prompt_tuning_method == "BDPL":
        average_theta = torch.FloatTensor([[1 / args.prompt_search_space] * args.prompt_search_space] * args.prompt_length)
    elif args.prompt_tuning_method == "prompt-tuning":
        average_theta = model.prompt_encoder.default.embedding.weight.clone().detach()
        

    # Start the training process. 
    for epoch in range(args.num_train_epochs):
        if args.FL_framework == "FedAvg":
            # training. 
            client_prompts_probs_list = []
            weight_list = []
            for client_idx in range(args.num_clients):
                # Each client train and update.  
                client_prompts_probs = client_list[client_idx].local_training(args, model, tokenizer, average_theta)
                client_prompts_probs_list.append(client_prompts_probs) #print("client_prompts_probs: \n", client_prompts_probs)
                # get the weight for averaging. 
                weight = client_list[client_idx].get_len_dataset() /len_entire_train_dataset
                weight_list.append(weight) #print("weight: \n", weight)
            # Fed Average.
            average_theta = sum(weight * tensor for weight, tensor in zip(weight_list, client_prompts_probs_list)) 
            if args.prompt_tuning_method == "prompt-tuning":
                    model.prompt_encoder.default.embedding.weight.data = average_theta

        elif args.FL_framework == "FedSeq":
            for client_idx in range(args.num_clients):
                average_theta = client_list[client_idx].local_training(args, model, tokenizer, average_theta)
                if args.prompt_tuning_method == "prompt-tuning":
                    model.prompt_encoder.default.embedding.weight.data = average_theta

        # Evaluation. 
        eval_result = evaluate(args, model, eval_dataloader, metric, ce_loss, config, accelerator, epoch, eval_results, prompts_probs=average_theta, prompt_length=prompt_length, tokenizer=tokenizer)
        #print(average_theta)

        if eval_result >= best_eval_result:
            best_eval_result = eval_result
            best_theta = average_theta.clone().detach()
            print(best_theta)
        if 'cuda' in str(args.device):
            torch.cuda.empty_cache()
        if train_api_request.count >= args.api_limit:
            break

    if args.prompt_tuning_method == "prompt-tuning":
        model.prompt_encoder.default.embedding.weight.data = best_theta

    test(args, model, test_dataloader, metric, accelerator, epoch, test_results, prompts_probs=best_theta, prompt_length=prompt_length, tokenizer=tokenizer, test_dataloader_mm=test_dataloader_mm)
    print( f"The total API calls for training in all client is: {train_api_request.count}")
    # train(args, accelerator, label_to_id, tokenizer, config, model, prompt_length, hingeloss, ce_loss, train_dataset, eval_dataset, test_dataset, data_collator )
