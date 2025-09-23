#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

scals=(20 40)
TARGET_DIR="/data/home/jvnting/cnpo/meta-llama"
LLAMA_DIR="/data/home/jvnting/cnpo/meta-llama"
for scal in "${scals[@]}"; do
	output_dir="/data/home/jvnting/cnpo/models/pii_7B_test_target-scal-${scal}/"
	pii="/data/home/jvnting/cnpo/data/pii/scal-${scal}/raw/synthetic_data.json"

	echo "Finetuning on datataset: ${pii}"
	echo "Saving to path: ${output_dir}"
	python -u baselines/baselines/finetune.py \
		--model_dir $TARGET_DIR \
		--tokenizer_dir $LLAMA_DIR \
		--data_file $pii \
		--out_dir $output_dir \
		--max_len 512 \
		--epochs 10 \
		--lr '2e-5' \
		--per_device_batch_size 4 \
		--algo 'it' 
done
