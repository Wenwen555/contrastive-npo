
CORPUS='news'

FORGET="/mnt/wenjt5/muse/data/$CORPUS/raw/forget.txt"
RETAIN="/mnt/wenjt5/muse/data/$CORPUS/raw/retain1.txt"

TARGET_DIR='/mnt/wenjt5/muse/model/pythia/pythia-410m-news'
LLAMA_DIR='/mnt/wenjt5/muse/model/pythia/pythia-410m-news'

MAX_LEN=2048
EPOCHS=10
LR='1e-5'
PER_DEVICE_BATCH_SIZE=2 # 3 GPUs
FT_EPOCHS=10
FT_LR='1e-5'


for algo in 'cont_npo'; do
    export CUDA_VISIBLE_DEVICES=1
    python baselines/unlearn.py \
        --algo $algo \
        --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
        --data_file $FORGET --retain_data_file $RETAIN \
        --out_dir "./ckpt/$CORPUS/cont_npo_cosine_1" \
        --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
        --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
	--coeff_type 'cosine'
done
