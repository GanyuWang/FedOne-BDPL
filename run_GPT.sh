# # GPT-based experiments
python ./run_glue_discrete_GPT.py \
--task_name=mrpc \
--per_device_train_batch_size 4 \
--per_device_eval_batch_size 4 \
--k_shot 16 --prompt_learning_rate 2e-4 \
--sample_size 20 --prompt_length 20 \
--prompt_search_space 50 --num_train_epochs 10 \
--api_key sk-proj-WrRU8jylYAl2sSTAboxnT3BlbkFJ4oV7Eit1mDHolIutf5uQ