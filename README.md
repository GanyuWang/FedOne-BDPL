# FedOne: Query-Efficient Federated Learning for Black-box Discrete Prompt Learning



## Requirements

To set up the environment, follow these steps:

1. **Create a virtual environment** for example using anaconda: 

   ```bash
   conda create -n bdpl python=3.9.19 -y
   conda activate bdpl
   ```

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
   The main file, "run_glue_LLM_FL.py," implements the federated discrete black-box prompt learning framework. The clients doing prompt tuning with different approaches (BBT, BDPL, Gumbel_BDPL, PrefixTuning, PromptTuning) are defined in the folder "PromptTuningClient/*.py". 

2. Run GPT-3.5-turbo experiments:

   To run GPT-3.5-turbo experiments, execute the following script:
   ```bash
   bash run_GPT.sh
   ```

   Make sure to obtain your [OpenAI API Key](https://openai.com/api/) and add it to a `.env` file in your project directory. The content of `.env` file should look like this:

   ```plaintext
   OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxx_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
   ```

   The main file "run_glue_LLM_FL_GPT.py" implements the federated discrete black-box prompt learning framework with chatGPT. The clients doing black-box prompt prompt learning (BDPL, Gumbel-BDPL, NoPrompt) using GPT-3.5-turbo is defined in the folder "PromptTuningClient_GPT". The primary difference between the GPT experiment and the RoBERTa-base experiment lies in data preprocessing and the method by which each model derives the output logits.



## Important Command-Line Arguments

### General Training Arguments

* `--task_name`: Specifies the name of the GLUE task. Options include: `[mrpc, qnli, cola, rte]`.
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

* `--FL_framework`: Specifies the Federated Learning framework. Currently supported: `FedAvg`. Default is `FedAvg`.
* `--num_clients`: Total number of clients in the Federated Learning setup. Default is `10`.
* `--num_activated_clients`: Number of clients activated in each training round. Default is `10`.
* `--num_client_local_step`: Number of local update steps performed by each client. Default is `1000`.
* `--max_client_train_steps`: Maximum number of training steps a client can perform during one activation. Default is `8000`.
* `--dirichlet_alpha`: Dirichlet concentration parameter for non-IID data partitioning. A value of `-1` indicates IID partitioning. Default is `-1.0`.

### Prompt Tuning Method Arguments

* `--prompt_tuning_method`: Specifies the prompt tuning strategy. Supported options include: `BBT`, `BDPL`, `GumbelBDPL`, `prefix-tuning`, and `prompt-tuning`. Default is `BDPL`.

#### BDPL-Specific Arguments

* `--bdpl_gradient_method`: Specifies the method used to estimate gradients in BDPL. Options: `negative`, `zero`, `normalize`. Default is `negative`.

#### BBT-Specific Arguments

* `--bbt_d`: Dimensionality parameter for BBT. Default is `500`.
* `--bbt_sigma`: Standard deviation parameter for the CMA-ES optimizer in BBT. Default is `1.0`.
* `--bbt_population_size`: Population size used by the CMA-ES optimizer. Default is `20`.

#### Gumbel-Softmax Arguments (BDPL)

* `--tau`: Temperature parameter for Gumbel-Softmax. Default is `0.1`.

### Early Stopping

* `--early_stop`: Training stops when the validation performance reaches this threshold. Default is `-1.0`.

### Logging

* `--log_file_name`: File name or path for saving training logs. Default is `TempResult`.


## Datasts
   [GLUE benchmark](https://gluebenchmark.com/): MNLI, QQP, SST-2, MRPC, CoLA, QNLI, RTE
