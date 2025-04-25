#!/bin/bash
export CUDA_VISIBLE_DEVICES=3
# 配置参数
CORPUS='news'  # 可选项: 'pii' 或 'news' 或 'books'
VERSION=2
MAX_LEN=2048
EPOCHS=10
LR='1e-5'
PER_DEVICE_BATCH_SIZE=1 # 1 GPUs
# FT_EPOCHS=10
# FT_LR='1e-5'
ALGO_TYPE='npo'  # 改为大写并更名以避免冲突
NEG_SAMPLE_NUM=2
COEFF_TYPE='cosine'

# 根据 CORPUS 设置路径
if [ "$CORPUS" = "pii" ]; then
    # PII 数据配置
    DATA_DIR="/data/home/jvnting/cnpo/data/pii/scal-5/raw"
    FORGET_DATA="$DATA_DIR/forget_sample.json"
    OUT_DIR_BASE="/data/home/jvnting/cnpo/ckpt/pii/"
elif [ "$CORPUS" = "news" ]; then
    # News 数据配置
    DATA_DIR="/data/home/jvnting/cnpo/data/news/raw"
    FORGET_DATA="$DATA_DIR/forget.txt"
    RETAIN_DATA="$DATA_DIR/retain1.txt"
    OUT_DIR_BASE="/data/home/jvnting/cnpo/ckpt/news"
elif [ "$CORPUS" = "books" ]; then
    # Books 数据配置 (请补充实际路径)
    DATA_DIR="/data/home/jvnting/cnpo/data/books/raw"
    FORGET_DATA="$DATA_DIR/forget.txt"
    RETAIN_DATA="$DATA_DIR/retain1.txt"
    OUT_DIR_BASE="/data/home/jvnting/cnpo/ckpt/books"
else
    echo "Error: Unknown CORPUS '$CORPUS'. Must be 'pii' or 'news' or 'books'."
    exit 1
fi

# 模型目录配置
TARGET_DIR="/data/home/jvnting/cnpo/models/MUSE/MUSE-News"
LLAMA_DIR="/data/home/jvnting/cnpo/meta-llama"

if [ "$ALGO_TYPE" = "npo" ]; then
    # NPO算法组的执行代码
    ALGOS=('npo' 'npo_klr' 'npo_gdr')  # 算法列表
    
    echo "Running NPO algorithms..."
    for current_algo in "${ALGOS[@]}"; do
        out_dir="$OUT_DIR_BASE/${current_algo}_${NEG_SAMPLE_NUM}n_${COEFF_TYPE}_7B_${VERSION}"
        
        echo "=== Running $current_algo algorithm ==="
        
        # 基础命令部分
        CMD="python -u baselines/unlearn.py \
            --algo \"$current_algo\" \
            --model_dir \"$TARGET_DIR\" \
            --tokenizer_dir \"$LLAMA_DIR\" \
            --data_file \"$FORGET_DATA\" \
            --out_dir \"$out_dir\" \
            --max_len \"$MAX_LEN\" \
            --epochs \"$EPOCHS\" \
            --lr \"$LR\" \
            --per_device_batch_size \"$PER_DEVICE_BATCH_SIZE\" \
            --coeff_type \"$COEFF_TYPE\" \
            --neg_sample_num \"$NEG_SAMPLE_NUM\" \
            --version \"$VERSION\""
        
        # 根据算法类型添加额外参数
        if [ "$current_algo" = "npo_klr" ] || [ "$current_algo" = "npo_gdr" ]; then
            CMD="$CMD --retain_file \"$RETAIN_DATA\""
            echo "Adding retain data for KLR/GDR variant..."
        fi
        
        # 执行命令
        eval $CMD
        
        echo "Finished $current_algo. Results saved to $out_dir"
    done
fi

# 第二组实验: 使用遗忘数据和保留数据
if [ "$ALGO_TYPE" = "cnpo" ]; then
    ALGOS=('cont_npo' 'cont_npo_klr' 'cont_npo_gdr')  # 算法列表
    for current_algo in "${ALGOS[@]}"; do
        out_dir="./ckpt/$CORPUS/${current_algo}_${NEG_SAMPLE_NUM}n_${COEFF_TYPE}_7B_${VERSION}"
        python -u baselines/unlearn.py \
            --algo "$current_algo" \
            --model_dir "$TARGET_DIR" --tokenizer_dir "$LLAMA_DIR" \
            --data_file "$FORGET_DATA" \
            --retain_data_file "$RETAIN_DATA" \
            --out_dir "$out_dir" \
            --max_len "$MAX_LEN" --epochs "$EPOCHS" --lr "$LR" \
            --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
            --coeff_type "$COEFF_TYPE" \
            --neg_sample_num "$NEG_SAMPLE_NUM" \
            --version $VERSION
    done
fi

