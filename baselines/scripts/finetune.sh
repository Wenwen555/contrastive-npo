
python  baselines/baselines/finetune.py \
    --model_dir '/root/autodl-tmp/contrastive-npo/models/pythia/pythia-410m' \
    --tokenizer_dir '/root/autodl-tmp/contrastive-npo/models/pythia/pythia-410m' \
    --data_file '/root/autodl-tmp/contrastive-npo/data/news/raw/merged_data.txt' \
    --out_dir '/root/autodl-tmp/contrastive-npo/models/pythia/pythia-410m-news' \
    --max_len 2048 \
    --epochs 5 \
    --lr 1e-5 \
    --per_device_batch_size 4 \
