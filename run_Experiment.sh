# RoBERTa-based experiments
python ./run_glue_LLM_FL.py \
--task_name=mrpc \
--model_name_or_path roberta-base \
--per_device_train_batch_size 128 \
--per_device_eval_batch_size 16 \
--weight_decay=0.1 --seed=42 \
--k_shot 16 --prompt_learning_rate 1e-4 \
--sample_size 20 --prompt_length 10 \
--prompt_search_space 200 \
--api_limit 8000 --ce_loss True \
--num_train_epochs 10 \
--FL_framework FedAvg --num_clients 10 --num_activated_clients 5 --num_client_local_step 1 --max_client_train_steps 8000 \
--prompt_tuning_method BDPL \
--log_file_name ExperimentResult/FedAvg_BDPL_NC10_AC5_Epoch10

# # GPT-based experiments
# python ./run_glue_discrete_GPT.py \
# --task_name=mrpc \
# --per_device_train_batch_size 4 \
# --per_device_eval_batch_size 4 \
# --k_shot 16 --prompt_learning_rate 2e-4 \
# --sample_size 20 --prompt_length 20 \
# --prompt_search_space 50 --num_train_epochs 10 \
# --api_key [API_KEY]
