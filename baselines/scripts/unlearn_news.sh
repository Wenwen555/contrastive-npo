CORPUS='news'

FORGET="../data/$CORPUS/raw/forget.txt"
RETAIN="../data/$CORPUS/raw/retain1.txt"

TARGET_DIR='../MUSE-News_target'
#LLAMA_DIR='../meta-llama/Llama-2-7b-hf'
LLAMA_DIR = '../meta-llama'

MAX_LEN=2048
EPOCHS=10
LR='1e-5'
PER_DEVICE_BATCH_SIZE=2 # 2 GPUs
FT_EPOCHS=10
FT_LR='1e-5'


for algo in 'ga_klr' 'npo' 'npo_gdr' 'npo_klr'; do
    python baselines/unlearn.py \
        --algo $algo \
        --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
        --data_file $FORGET --retain_data_file $RETAIN \
        --out_dir "./ckpt/$CORPUS/$algo" \
        --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
        --per_device_batch_size $PER_DEVICE_BATCH_SIZE
done


python baselines/unlearn.py \
    --algo 'tv' \
    --model_dir $TARGET_DIR --tokenizer_dir $LLAMA_DIR \
    --data_file $FORGET --retain_data_file $RETAIN \
    --out_dir "./ckpt/$CORPUS/tv" \
    --max_len $MAX_LEN --epochs $FT_EPOCHS --lr $FT_LR \
    --per_device_batch_size $PER_DEVICE_BATCH_SIZE \
    --alpha 5.0
