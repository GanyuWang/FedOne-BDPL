# Tuning command. 
ac=1
task_name=cola
prompt_tuning_method=BDPL
prompt_learning_rate=3e-5
prompt_length=20
bbt_population_size=200
early_stop=77e-2

# Repeat experiment
for seed in 101
    do
    echo activated_client_${ac}_seed_${seed}
    CUDA_VISIBLE_DEVICES=2 python ./run_glue_LLM_FL.py \
        --task_name ${task_name} \
        --prompt_tuning_method ${prompt_tuning_method} \
        --bdpl_gradient_method zero \
        --model_name_or_path roberta-large \
        --per_device_train_batch_size 128 \
        --per_device_eval_batch_size 16 \
        --weight_decay=0.01 --seed=$seed \
        --k_shot 16 --prompt_learning_rate ${prompt_learning_rate} \
        --sample_size 20 --prompt_length ${prompt_length} \
        --prompt_search_space 200 \
        --api_limit 80000 --ce_loss True \
        --bbt_population_size ${bbt_population_size} \
        --num_train_epochs 100 \
        --FL_framework FedAvg --num_clients 100 --num_activated_clients ${ac} --num_client_local_step 1 --max_client_train_steps 8000 \
        --early_stop ${early_stop} \
        --log_file_name TempResult/${task_name}_${prompt_tuning_method}_ps${bbt_population_size}_lr${prompt_learning_rate}_pl${prompt_length}_ac${ac}_es${early_stop}_seed${seed}
done