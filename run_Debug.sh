ac=1
task_name=cola
prompt_tuning_method=BDPL
prompt_learning_rate=1e-4
prompt_length=20
bbt_population_size=200
early_stop=90e-2


seed=49
echo ${task_name}_${prompt_tuning_method}_
CUDA_VISIBLE_DEVICES=0 python ./run_glue_LLM_FL.py \
    --task_name=${task_name} \
    --prompt_tuning_method ${prompt_tuning_method} \
    --bdpl_gradient_method zero \
    --model_name_or_path roberta-base \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 16 \
    --weight_decay=0 --seed=$seed \
    --k_shot 16 --prompt_learning_rate ${prompt_learning_rate} \
    --sample_size 20 --prompt_length ${prompt_length} \
    --prompt_search_space 100 \
    --api_limit 8000 --ce_loss True \
    --bbt_population_size ${bbt_population_size} \
    --num_train_epochs 10 \
    --tau 0.1 \
    --FL_framework FedAvg --num_clients 100 --num_activated_clients ${ac} --num_client_local_step 10 --max_client_train_steps 8000 \
    --early_stop ${early_stop} \
    --log_file_name TempResult 
# When runing, skip should all be false. 

