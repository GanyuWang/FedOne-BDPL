ac=1
task_name=cola
prompt_tuning_method=GumbelBDPL
prompt_learning_rate=1e-3
prompt_length=20
bbt_population_size=200
early_stop=90e-2
seed=51


echo ExperimentResult_GPT/${task_name}_${prompt_tuning_method}_seed${seed}
CUDA_VISIBLE_DEVICES=0 python ./run_glue_LLM_FL_GPT.py \
    --task_name=${task_name} \
    --prompt_tuning_method ${prompt_tuning_method} \
    --bdpl_gradient_method zero \
    --model_name_or_path gpt-3.5-turbo-1106 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --weight_decay=0.01 --seed=$seed \
    --k_shot 16 --prompt_learning_rate ${prompt_learning_rate} \
    --sample_size 3 --prompt_length ${prompt_length} \
    --prompt_search_space 200 \
    --api_limit 8000 --ce_loss True \
    --bbt_population_size ${bbt_population_size} \
    --num_train_epochs 5 \
    --max_tokens 12 --top_logprob 3\
    --FL_framework FedAvg --num_clients 100 --num_activated_clients ${ac} --num_client_local_step 1 --max_client_train_steps 8000 \
    --early_stop ${early_stop} \
    --trial --train_trial_step 2 --eval_trial_step 7 --test_trial_step 10\
    --log_file_name ExperimentResult_GPT/${task_name}_${prompt_tuning_method}_seed${seed}
echo ExperimentResult_GPT/${task_name}_${prompt_tuning_method}_seed${seed}

# --skip_training\
# 0125
# 1106