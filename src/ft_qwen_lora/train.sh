lora_rank=8
#lora_trainable="query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
model_name_or_path="/root/autodl-fs/data2/root/.cache/modelscope/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"   # LLM底座模型路径，或者是huggingface hub上的模型名称
your_data_path="./datasets/trains/"  # 填入数据集所在的文件夹路径
your_data_path="./datasets/toys/"  # 填入数据集所在的文件夹路径
your_checkpoint_path="/root/autodl-fs/data2/models/Qwen2.5-7B/"  # 填入用来存储模型的路径

peft_path=""  # 如果之前训练过，且存储了peft权重，则设置为peft权重的文件夹路径
resume_from_checkpoint=""

CUDA_VISIBLE_DEVICES=0 python src/ft_qwen_lora/main.py \
    --do_train \
    --do_eval \
    --train_file $your_data_path/train.jsonl \
    --validation_file $your_data_path/dev.jsonl \
    --cache_dir $your_checkpoint_path/TCM-NER-Qwen25-7B-lora-$LR \
    --prompt_column input \
    --response_column target \
    --model_name_or_path $model_name_or_path \
    --output_dir $your_checkpoint_path/TCM-NER-Qwen25-7B-lora-$LR \
    --overwrite_output_dir \
    --max_source_length 128 \
    --max_target_length 64 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_steps 1500 \
    --logging_steps 10 \
    --save_steps 100 \
    --learning_rate $LR \
    --lora_rank ${lora_rank} \
    --modules_to_save ${modules_to_save} \
    --lora_dropout ${lora_dropout} \
    --fp16 \
    --preprocessing_num_workers 12 \
    --report_to wandb \
#    --trainable ${lora_trainable} \
