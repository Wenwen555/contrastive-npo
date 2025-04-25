#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# 清除可能冲突的变量
unset LOCAL_RANK WORLD_SIZE

TARGET_DIR='/data/home/jvnting/cnpo/meta-llama'
LLAMA_DIR='/data/home/jvnting/cnpo/meta-llama'
output_dir='/data/home/jvnting/cnpo/models/pii_7B_without_target/'
pii='/data/home/jvnting/cnpo/data/pii-data/train_synthetic_replacements-1k.pkl'

torchrun --nproc_per_node=1 --master_port=29505 baselines/baselines/finetune_ac.py \
	--model_dir $TARGET_DIR \
	--tokenizer_dir $LLAMA_DIR \
	--data_file $pii \
	--out_dir $output_dir \
	--max_len 2048 \
	--epochs 5 \
	--lr '1e-5' \
	--per_device_batch_size 1
# # 使用 torchrun 启动（注意参数位置）
# accelerate launch baselines/baselines/finetune_ac.py \
#     --model_dir "$TARGET_DIR" \
#     --tokenizer_dir "$LLAMA_DIR" \
#     --data_file "$pii" \
#     --out_dir "$output_dir" \
#     --max_len 2048 \
#     --epochs 5 \
#     --lr '1e-5' \
#     --per_device_batch_size 1

# TARGET_DIR='/data/home/jvnting/cnpo/meta-llama'
# LLAMA_DIR='/data/home/jvnting/cnpo/meta-llama'
# output_dir='/data/home/jvnting/cnpo/models/pii_7B_without_target/'

# pii='/data/home/jvnting/cnpo/data/pii-data/train_synthetic_replacements-1k.pkl'

# python -u baselines/baselines/finetune_ac.py \
# 	--model_dir $TARGET_DIR \
# 	--tokenizer_dir $LLAMA_DIR \
# 	--data_file $pii \
# 	--out_dir $output_dir \
# 	--max_len 2048 \
# 	--epochs 5 \
# 	--lr '1e-5' \
# 	--per_device_batch_size 1
