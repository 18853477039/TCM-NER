lora_rank=8
#lora_trainable="query_key_value,dense,dense_h_to_4h,dense_4h_to_h"
modules_to_save="null"
lora_dropout=0.1
LR=2e-4
model_name_or_path="/root/autodl-fs/data2/root/.cache/modelscope/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28"   # LLM底座模型路径，或者是huggingface hub上的模型名称
your_data_path="./datasets/sft/"  # 填入数据集所在的文件夹路径
your_data_path="./datasets/toys/"  # 填入数据集所在的文件夹路径

CHECKPOINT="/root/autodl-fs/data2/models/Qwen2.5-7B/TCM-NER-Qwen25-7B-lora-2e-4"   # 填入用来存储模型的文件夹路径

STEP=3500    # 用来评估的模型checkpoint是训练了多少步

CUDA_VISIBLE_DEVICES=0 python src/ft_qwen_lora/main.py \
    --do_eval \
    --do_predict \
    --validation_file $your_data_path/dev.jsonl \
    --test_file $your_data_path/test.jsonl \
    --cache_dir $your_data_path \
    --overwrite_cache \
    --prompt_column input \
    --response_column target \
    --model_name_or_path $model_name_or_path \
    --peft_path $CHECKPOINT/checkpoint-$STEP \
    --output_dir $CHECKPOINT/checkpoint-$STEP \
    --overwrite_output_dir \
    --max_source_length 128 \
    --max_target_length 64 \
    --per_device_eval_batch_size 12 \
    --preprocessing_num_workers 12 \
    --report_to wandb


