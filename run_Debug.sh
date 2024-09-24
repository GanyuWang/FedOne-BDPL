ac=1
task_name=rte
prompt_tuning_method=GumbelBDPL
prompt_learning_rate=3e-5
prompt_length=20
bbt_population_size=200
early_stop=90e-2


seed=49
echo activated_client_${ac}_seed_${seed}
CUDA_VISIBLE_DEVICES=0 python ./run_glue_LLM_FL_GPT.py \
    --task_name=${task_name} \
    --prompt_tuning_method ${prompt_tuning_method} \
    --bdpl_gradient_method zero \
    --model_name_or_path gpt-3.5-turbo \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --weight_decay=0.01 --seed=$seed \
    --k_shot 16 --prompt_learning_rate ${prompt_learning_rate} \
    --sample_size 5 --prompt_length ${prompt_length} \
    --prompt_search_space 200 \
    --api_limit 8000 --ce_loss True \
    --bbt_population_size ${bbt_population_size} \
    --num_train_epochs 1 \
    --max_tokens 12 --top_logprob 3\
    --FL_framework FedAvg --num_clients 100 --num_activated_clients ${ac} --num_client_local_step 1 --max_client_train_steps 8000 \
    --early_stop ${early_stop} \
    --trial --trial_step 1\
    --log_file_name TempResult 
# When runing, skip should all be false. 

