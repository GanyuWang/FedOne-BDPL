
## Project Overview

This repository contains the official code for our ICML 2025 paper:

**[FedOne: Query-Efficient Federated Learning for Black-box Discrete Prompt Learning](https://openreview.net/forum?id=QwTDQXllam)**

This work proposes a novel federated framework designed to minimize query costs to cloud-based LLM in black-box discrete prompt learning scenarios. 

The implementation builds upon and extends the codebase from [Black-Box-Prompt-Learning](https://github.com/shizhediao/Black-Box-Prompt-Learning), adapting it to the **federated learning** setting with additional components for client coordination, efficient prompt optimization. 


### Main Files

* **RoBERTa-based Experiments**:

  * `preprocess.py`: Performs data loading and preprocessing for RoBERTa tasks.
  * `run_glue_LLM_FL.py`: Implements the federated learning framework for RoBERTa-based prompt tuning.
  * `PromptTuningClient/*.py`: Contains client-side implementations of various white-box prompt tuning methods, including BBT, BDPL, Gumbel-BDPL, Prefix-Tuning, and Prompt-Tuning.

* **OpenAI API-based Experiments (GPT models)**:

  * `preprocess_GPT.py`: Handles preprocessing tailored to GPT-based experiments using the [OpenAI API](https://platform.openai.com/docs/overview).
  * `run_glue_LLM_FL_GPT.py`: Implements the federated learning workflow for black-box prompt tuning with GPT models.
  * `PromptTuningClient_GPT/*.py`: Includes client-side implementations for black-box prompt learning methods such as BDPL, Gumbel-BDPL, and NoPrompt.



## Requirements

To set up the environment, follow these steps:

1. **Create a virtual environment** for example using anaconda: 

   ```bash
   conda create -n bdpl python=3.9.19 -y
   conda activate bdpl
   ```
   We used python version (3.9.19). 

2. **Install required packages:**

   Use the ``requirement.txt'' to install the required libraries.

   ```bash
   pip install -r requirements.txt
   ```


## Quick Start
1. For RoBERTa-large experiments, run the scripts via 
   ```bash
   bash run_Experiment.sh
   ```

2. Run GPT-3.5-turbo experiments:

   To run GPT-3.5-turbo experiments, execute the following script:
   ```bash
   bash run_GPT.sh
   ```

   Make sure to obtain your [OpenAI API Project Key](https://openai.com/api/) and add it to a `.env` file in your project directory. The content of `.env` file should look like this:

   ```plaintext
   OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxx_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```


## Important Command-Line Arguments

### General Training Arguments

* `--task_name`: Specifies the name of the GLUE task. Options include: `[mnli, qqp, sst2, mrpc, cola, qnli, rte]`.
* `--file_name`: Indicates the name of the domain-specific dataset. Options include: `[CI, SE, RCT, HP]`.
* `--low_resource`: Enables low-resource training mode.
* `--ce_loss`: Specifies whether to use cross-entropy loss. If set to `False`, hinge loss will be used. Default is `True`.
* `--sample_size`: Defines the number of samples per batch. This parameter is critical for controlling resource usage. Default is `20`.
* `--prompt_length`: Sets the length of the prompt tokens. Default is `6`.
* `--prompt_learning_rate`: Learning rate used for prompt tuning. Default is `5e-5`.
* `--prompt_search_space`: The size of the search space for prompt optimization. Default is `20`.
* `--num_train_epochs`: Total number of training epochs to perform. Default is `30`.
* `--ckpt_path`: Path for saving model checkpoints. Default is `./ckpts`.
* `--margin`: Margin used in the loss function. Default is `1.0`.
* `--trial`: If enabled, denotes a trial run for debugging or exploratory experiments.
* `--use_wandb`: Specifies whether to use Weights & Biases for experiment tracking. Default is `False`.
* `--cuda`: The ID of the CUDA device to use. Default is `0`.
* `--max_length`: The maximum length of input sequences after tokenization. Longer sequences are truncated. Default is `450`.
* `--pad_to_max_length`: If enabled, all sequences are padded to `max_length`. Otherwise, dynamic padding is used.
* `--per_device_train_batch_size`: Batch size per device during training. Default is `128`.
* `--per_device_eval_batch_size`: Batch size per device during evaluation. Default is `32`.
* `--model_name_or_path`: Path to a pretrained model or its identifier from Hugging Face. Default is `'roberta-large'`.
* `--use_slow_tokenizer`: If enabled, uses the slower tokenizer implementation not backed by the Hugging Face Tokenizers library.
* `--weight_decay`: Weight decay coefficient for regularization. Default is `0.1`.
* `--max_train_steps`: If specified, overrides the number of training epochs.
* `--gradient_accumulation_steps`: Number of steps to accumulate gradients before performing a backward pass. Default is `1`.
* `--lr_scheduler_type`: Specifies the learning rate scheduler type. Options include: `linear`, `cosine`, `cosine_with_restarts`, `polynomial`, `constant`, and `constant_with_warmup`. Default is `linear`.
* `--num_warmup_steps`: Number of warm-up steps for the learning rate scheduler. Default is `100`.
* `--output_dir`: Directory for saving the final trained model.
* `--seed`: Random seed for reproducibility. Default is `42`.
* `--k_shot`: Number of examples per class for few-shot learning. A value of `-1` denotes full supervision. Default is `-1`.
* `--use_ngram`: Indicates whether to use n-gram features. Default is `True`.
* `--api_limit`: Maximum number of API requests allowed. Default is `8000`.

### Federated Learning Arguments

* `--FL_framework`: Specifies the Federated Learning framework. Currently supported: `FedAvg`.
* `--num_clients`: Total number of clients in the Federated Learning setup. Default is `10`.
* `--num_activated_clients`: Number of clients activated in each training round. Default is `10`.
* `--num_client_local_step`: Number of local update steps performed by each client. Default is `1000`.
* `--max_client_train_steps`: Maximum number of training steps a client can perform during one activation. Default is `8000`.
* `--dirichlet_alpha`: Dirichlet concentration parameter for non-IID data partitioning. A value of `-1` indicates IID partitioning, other value all using Dirichlet partition. Default is `-1.0`.

### Prompt Tuning Method Arguments

* `--prompt_tuning_method`: Specifies the prompt tuning strategy. Supported options include: `BBT`, `BDPL`, `GumbelBDPL`, `prefix-tuning`, and `prompt-tuning`. Default is `BDPL`.

#### BBT-Specific Arguments

* `--bbt_d`: Dimensionality parameter for BBT. Default is `500`.
* `--bbt_sigma`: Standard deviation parameter for the CMA-ES optimizer in BBT. Default is `1.0`.
* `--bbt_population_size`: Population size used by the CMA-ES optimizer. Default is `200`.

#### Gumbel-Softmax Arguments (BDPL)

* `--tau`: Temperature parameter for Gumbel-Softmax. Default is `0.1`.

### Early Stopping

* `--early_stop`: Training will stop once the validation metric meets or exceeds this value. If set to a value less than 0, early stopping is disabled and training will proceed for the full number of epochs. Default is `-1.0`.

### Logging

* `--log_file_name`: Specifies the **file path** for saving training logs. The default value is `TempResult`. When the path starts with `TempResult`, the log file can be overwritten in subsequent runs. **For all other values, the system will prevent overwriting an existing log file to avoid accidental loss of results**. Upon completion of training, the final row of the log file records the test result. When conducting experiments, it is recommended to create a dedicated folder for each run to organize logs. 



## Datasts
   [GLUE benchmark](https://gluebenchmark.com/): MNLI, QQP, SST-2, MRPC, CoLA, QNLI, RTE


