# FedOne: Query-Efficient Federated Learning for Black-box Discrete Prompt Learning



## Requirements

To set up the environment, follow these steps:

1. **Create a virtual environment** for example using anaconda: 
   ```bash
   conda create -n bdpl python=3.9 -y
   conda activate bdpl
   ```

2. **Install required packages:**

   Run the `install.sh` script to install all necessary packages with `pip`:
   ```bash
   bash install.sh
   ```


## Quick Start
1. For RoBERTa-based experiments, run the scripts via 
   ```bash
   bash run_Experiment.sh
   ```
   The main file, "run_glue_LLM_FL.py," implements the federated discrete black-box prompt learning framework. The clients doing prompt tuning with different approaches (BBT, BDPL, Gumbel_BDPL, PrefixTuning, PromptTuning) are defined in the folder "PromptTuningClient". 

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

3. Important arguments:
   
   Training Arguments

   * `--task_name`: The name of a glue task. choices = `[mrpc, qnli, cola, rte]`.
   * `--file_name`: The name of the domain-specific task. choices = `[CI, SE, RCT, HP]`.
   * `--ce_loss`: if true, use cross-entropy loss. otherwise, use hinge loss.
   * `--prompt_length`: number of prompt.
   * `--k_shot`: number of shots.
   * `--api_key`: GPT-3 openai access key.

   Federated Learning Arguments

   * `--FL_framework`: Specifies the Federated Learning framework. Choices: `FedAvg`, `FedSeq`. Default: `FedAvg`.
   * `--num_clients`: The number of clients in Federated Learning. Default: `10`.
   * `--num_activated_clients`: The number of activated clients in each epoch of Federated Learning. Default: `10`.
   * `--num_client_local_step`: The number of local update epochs for each client in Federated Learning. Default: `1000`.
   * `--max_client_train_steps`: The maximum number of local iterations for a client per activation. Default: `8000`.

   Prompt Tuning Method Arguments

   * `--prompt_tuning_method`: Specifies the tuning method. Choices: `BBT`, `BDPL`, `GumbelBDPL`, `prefix-tuning`, `prompt-tuning`. Default: `BDPL`.

   BDPL Arguments

   * `--bdpl_gradient_method`: The way to estimate the gradient for BDPL. Choices: `negative`, `zero`, `normalize`. Default: `negative`.

   BBT Parameters

   * `--bbt_d`: The `d` parameter for BBT. Default: `500`.
   * `--bbt_sigma`: The sigma for CMA-ES in BBT. Default: `1.0`.
   * `--bbt_population_size`: The population size for CMA-ES in BBT. Default: `20`.

   BDPL Gumbel Softmax

   * `--tau`: The temperature of the Gumbel-Softmax. Default: `0.1`.

   Early Stopping

   * `--early_stop`: Stops training when the validation result reaches this threshold. Default: `-1.0`.

   Log File

   * `--log_file_name`: The file path for saving logs. Default: `TempResult`.

## Datasts
   [GLUE benchmark](https://gluebenchmark.com/): MNLI, QQP, SST-2, MRPC, CoLA, QNLI, RTE
