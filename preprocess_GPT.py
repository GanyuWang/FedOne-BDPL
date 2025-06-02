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
#from transformers.models.roberta.configuration_roberta import RobertaConfig
#from transformers.models.roberta.modeling_roberta import RobertaClassificationHead, RobertaForMaskedLM
#from transformers import RobertaModel
from torch.nn import CrossEntropyLoss
from loss import *
import wandb
from peft import get_peft_config, get_peft_model,  TaskType, PeftType
#from peft import PromptTuningInit, PromptTuningConfig, PrefixTuningConfig, PromptEncoderConfig

import pandas as pd
import openai 
import sys



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
    'imdb': {0: ' terrible', 1: ' great'},
    'cr': {0: ' terrible', 1: ' great'},
}

TEMPLATE_CONFIG = {
    "mnli": "Reply me with one word: 'yes', 'maybe', or 'no':\n",
    "qqp":  "Reply me with one word: 'yes' or 'no':\n",
    "sst2": "Reply me with one word: 'great' or 'terrible':\n",
    "mrpc": "Reply me with one word: 'yes' or 'no':\n",
    "cola": "Reply me with one word: 'yes' or 'no':\n",
    "wnli": "Reply me with one word: 'yes' or 'no':\n",
    "qnli": "Reply me with one word: 'yes' or 'no':\n",
    "rte":  "Reply me with one word: 'yes' or 'no':\n",
    "CI": " What is the intent?",
    "SE": " What is the relation?",
    "RCT": " What is the role?",
    "HP": " Helpful?",
    "imdb": " It was .",
    "cr": " It was ",
}

# With correct prompt. 
# TEMPLATE_CONFIG = {
#     "mnli": "Entailment? Reply me with one word: 'yes', 'maybe', or 'no':\n",
#     "qqp": "Equivalent? Reply me with one word: 'yes' or 'no':\n",
#     "sst2": "What is the sentiment? Reply me with one word: 'great' or 'terrible':\n",
#     "mrpc": "Equivalent? Reply me with one word: 'yes' or 'no':\n",
#     "cola": "Correct? Reply me with one word: 'yes' or 'no':\n",
#     "wnli": "What is the relation? Reply me with one word: 'yes' or 'no':\n",
#     "qnli": "Entailment? Reply me with one word: 'yes' or 'no':\n",
#     "rte": "Entailment? Reply me with one word: 'yes' or 'no':\n",
#     "CI": " What is the intent?",
#     "SE": " What is the relation?",
#     "RCT": " What is the role?",
#     "HP": " Helpful?",
#     "imdb": " It was .",
#     "cr": " It was ",
# }




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
        with open("./pmi/" + "pmi_" + args.file_name.lower() + "_gpt" + ".txt",'r') as f:
            for line in f:
                result.append(line.strip('\n'))
    elif args.task_name:
        with open("./pmi/" + "pmi_" + args.task_name.lower() + "_gpt" + ".txt",'r') as f:
            for line in f:
                result.append(line.strip('\n'))
    unique = []
    [unique.append(i) for i in result if not i in unique]
    ngram_index_list = list(unique)
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


def create_batches(dataset, batch_size=1, shuffle=False):
    if isinstance(dataset, dict):
        dataset_dict = dataset
    else:
        dataset_dict = {'input': [], 'labels': []}
        dataset_dict['input'] = dataset['input']
        dataset_dict['labels'] = dataset['labels']
    
    if shuffle:
        dataset_dict = pd.DataFrame(dataset_dict)
        dataset_dict = dataset_dict.sample(frac=1)
        dataset_dict = dataset_dict.to_dict(orient='list')
    batches = {'sentence': [], 'labels':[]}  # 这个要重做。
    for i in range(0,len(dataset_dict['input']),batch_size):
        batches['sentence'].append(dataset_dict['input'][i: i + batch_size])
        batches['labels'].append(dataset_dict['labels'][i: i + batch_size])
    return batches

import openai
from openai import OpenAI
# @counter
class CompleteGPT():
    def __init__(self):
        self.client = OpenAI(
            #organization='org-xxxxxxxxxxxxxxxxxxx',
            #project='proj_xxxxxxxxxxxxxxxxxxxxxxx',
        )
        self.wait_time = 1

    def complete_gpt3(self, prompt, max_tokens, model_name, n=1, top_logprob=5):
        response = None
        received = False
        
        while not received:
            try:
                response = self.client.chat.completions.create(
                            model=model_name,
                            messages=prompt,
                            logprobs=True,
                            max_tokens=max_tokens,
                            n=n,
                            top_logprobs=top_logprob,
                            temperature=0.)
                received = True
                self.wait_time = self.wait_time-1
                if self.wait_time <= 0: self.wait_time = 0.6
                time.sleep(self.wait_time)
            except openai.RateLimitError as e:
                print("An rate limit error occurred:" ) # An error occurred: name 'x' is not defined
                self.wait_time += 10
                time.sleep(self.wait_time)
        return response
        """
        response = None
        received = False
        while not received:
            try:
                response = self.client.chat.completions.create(
                    model=model_name,
                    messages=prompt,
                    logprobs=True,
                    max_tokens=l,
                    n=n,
                )
                received = True
            except:
                error = sys.exc_info()[0]
                if error == openai.BadRequestError:
                    print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                    assert False
                print("API error:", error)
                time.sleep(1)
        return response
        """
    @counter
    def train_api_request(self, prompt, max_tokens, model_name, n=1, top_logprob=1):
        response=self.complete_gpt3(prompt, max_tokens, model_name, n=n, top_logprob=top_logprob)
        return response

    class ApiCallLimitError(Exception):
        pass

    def get_regular_label_probs(self, responses, batch, labels, args, if_null=True, split="train"):
        assert len(responses.choice) == len(batch)
        label_probs = torch.zeros([len(responses.choice), 1, len(labels)])
        all_missing_positions = []
        for a, ans in enumerate(responses.choice):
            for l, label in enumerate(labels):
                if label == ans.logprobs.content[0].token:
                    label_probs[a,:,l] = np.exp(ans.logprobs.content[0].logprob)
                else:
                    position = (a, l)
                    all_missing_positions.append(position)
                    
        if len(all_missing_positions) > 0:
            all_additional_prompts = []
            for position in all_missing_positions:
                which_sentence, which_label = position
                missing_prompt = batch[which_sentence] + labels[which_label]
                all_additional_prompts.append(missing_prompt)
            additional_dataset = {'input': all_additional_prompts, 'labels': all_missing_positions}
            batches = create_batches(additional_dataset, batch_size=len(batch[0]))
            for m, missing_batch in enumerate(batches):
                if split == "train":
                    missing_response = self.train_api_request(missing_batch, l=0, model_name=args.model_name_or_path)
                else:
                    missing_response = self.complete_gpt3(missing_batch, l=0, model_name=args.model_name_or_path)
                for idx, missing_ans in enumerate(missing_response['choices']):
                    which_sentence, which_label = batches['labels'][m][idx]
                    label_probs[which_sentence,:,which_label] = np.exp(missing_ans['logprobs']['token_logprobs'][-1])
        assert (label_probs > 0).all(), "all should be populated with non-zero value"
                
        if if_null:
            return label_probs
        label_probs = label_probs / torch.sum(label_probs, dim=2, keepdim=True)
        return label_probs
    
    def get_label_prob(self, response, chat_obj, label_keys, args, prob_if_label_not_found=0.01):
        labels_prob = torch.zeros(len(label_keys))
        print(chat_obj)
        print(response.choices[0].message.content)
        for label_index, label in enumerate(label_keys):  
            found_the_label = False
            #print(f"finding {label}.", end=" ")
            for j in range(len(response.choices[0].logprobs.content[0].top_logprobs)): # for i in range(len(response.choices[0].logprobs.content)):  J first because, we want the top prob first. 
                for i in range(len(response.choices[0].logprobs.content)): # for j in range(len(response.choices[0].logprobs.content[i].top_logprobs)):
                    if label[1:].startswith(response.choices[0].logprobs.content[i].top_logprobs[j].token.lower()):   # This is tricky, the token for "terrible" is "ter" and "rible". 2) " great"[1:] = "great"
                        prob = np.exp(response.choices[0].logprobs.content[i].top_logprobs[j].logprob)
                        labels_prob[label_index] = prob
                        found_the_label = True
                        #print(f"YYY<{label}>YYY, [{response.choices[0].logprobs.content[i].top_logprobs[j].token}], i={i}, j={j}, prob={prob} ", end=" ")
                    if found_the_label: break
                if found_the_label: break
            # be careful about the indent. 
            if not found_the_label:
                #print(f"xxx<{label}>xxx", end=" ")
                labels_prob[label_index] = prob_if_label_not_found # small probl
  
        """
        if label in response.choices[0].logprobs.content[0].token:
            label_prob = np.exp(response.choices[0].logprobs.content[0].logprob)
            return label_prob
        else:
            missing_response = self.train_api_request(chat_obj, l=100, model_name=args.model_name_or_path, n=1, top_logprob=1)
            print("the missing label is : ", label)
            for i in range(20):
                if label == missing_response.choices[0].logprobs.content[0].top_logprobs[i].token:
                    label_prob = np.exp(response.logprobs.content[0].logprob)
                    return label_prob
        """
        print("\n", labels_prob)
        return labels_prob # a small label. 


class ApiCallLimitError(Exception):
    pass

def prepare_and_load_dataset(args):
    
    ngram_list = pmi(args)

    assert args.task_name != 'stsb'
    ce_loss_string = 'True' if args.ce_loss else 'False'

    # specify a unique experiment_id for load_metric() otherwise will cause ERROR when having multiple run on a same server!
    task_name = args.task_name if args.task_name else args.train_file
    args.unique_task_name = task_name.replace("/", ".")
    args.experiment_id = task_name + str(args.prompt_length) + str(args.prompt_learning_rate) +\
                         str(args.learning_rate) + str(args.num_train_epochs) \
                         + str(args.seed) + str(args.prompt_search_space) + str(args.std) + ce_loss_string

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
        id_to_label = LABEL_CONVERT[args.task_name]
    elif args.file_name:
        label_to_id = LABEL2ID_CONFIG[args.file_name]
        id_to_label = LABEL_CONVERT[args.file_name]
    args.num_labels = len(label_to_id)

    # Load pretrained model and tokenizer
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=not args.use_slow_tokenizer)
    args.device = torch.device("cuda", args.cuda)

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

    #padding = "max_length" if args.pad_to_max_length else False

    
    def preprocess_function(examples):
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

        result= {'input':[]}

        for i in range(len(examples[sentence1_key])):
            if sentence2_key is None:
                ori_sent_id = tokenizer.tokenize(examples[sentence1_key][i])[:400]
                new_sent = tokenizer.convert_tokens_to_string(ori_sent_id)
                result["input"].append( template_cfg + new_sent + "\n")   # Specific for GPT 3.5.  Delete output. 
            else:
                result["input"].append( template_cfg + f' input sentence one: \"{examples[sentence1_key][i]}\" ' + f' sentence two: \"{examples[sentence2_key][i]}\" \n')

        if args.task_name or args.file_name in DOMAIN_DATASET:
            result['labels'] = [id_to_label[x] for x in examples["label"]]
        else:
            result['labels'] = examples["label"]

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
            elif args.task_name == 'qqp':
                raw_valid_dataset_split = raw_datasets["validation"].train_test_split(test_size=0.025)
                raw_test_dataset = raw_valid_dataset_split['test']
                test_dataset = raw_test_dataset.map(
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
    
    train_batches = create_batches(train_dataset, batch_size=args.per_device_train_batch_size, shuffle=True)
    eval_batches = create_batches(eval_dataset, batch_size=args.per_device_eval_batch_size)
    test_batches = create_batches(test_dataset, batch_size=args.per_device_eval_batch_size)
    if args.task_name == 'mnli':
        test_batches_mm = create_batches(test_dataset_mm, batch_size=args.per_device_eval_batch_size)
        test_batches_mm = accelerator.prepare(test_batches_mm)
    else:
        test_batches_mm = None
    eval_batches, test_batches = accelerator.prepare(eval_batches, test_batches)


    # Note -> the training dataloader needs to be prepared before we grab his length below (cause its length will be shorter in multiprocess)
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_batches['sentence']) / args.gradient_accumulation_steps) # 106
    args.max_train_steps = args.num_train_epochs * (num_update_steps_per_epoch)
    # Get the metric function
    if args.task_name is not None:
        metric = load_metric("glue", args.task_name, experiment_id=args.experiment_id)
    elif args.file_name in DOMAIN_DATASET:
        metric = load_metric('f1', args.experiment_id)
    else:
        metric = load_metric('accuracy', args.experiment_id)

    return (accelerator, label_to_id, tokenizer, prompt_length, metric, ngram_list), \
           (hingeloss, ce_loss), \
           (train_dataset, eval_dataset, test_dataset), \
           (train_batches, eval_batches, test_batches, test_batches_mm) 





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
        if log_file_name.startswith("TempResult"):
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


