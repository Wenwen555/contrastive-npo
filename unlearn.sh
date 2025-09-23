#!/bin/bash
export CUDA_VISIBLE_DEVICES=2

# 配置参数
VERSION=3
EPOCHS=10
LR='1e-5'
PER_DEVICE_BATCH_SIZE=1 # 1 GPUs
scals=(10)

# 下次unlearning计划，在beta=1下跑ga和simnpo，并且把各算法的klr补充
# 挑选SOTA的CNPO
# 定义要遍历的corpus和算法
CORPORA=("pii")
NEG_SAMPLE_NUMS=(2)
# ALGO_TYPES=("ga" "npo" "simnpo" "cnpo")
# "ga" "npo" "simnpo" 
ALGO_TYPES=("cnpo")

# 遍历corpus
for CORPUS in "${CORPORA[@]}"; do
    for scal in "${scals[@]}"; do
        # 根据CORPUS设置路径  
        if [ "$CORPUS" = "news" ]; then
            # News数据配置
            DATA_DIR="/data/home/jvnting/cnpo/data/news/raw"
            FORGET_DATA="$DATA_DIR/forget.txt"
            RETAIN_DATA="$DATA_DIR/retain1.txt"
            OUT_DIR_BASE="/data/home/jvnting/cnpo/ckpt/news/final"
            COEFF_TYPE='cosine'
            TARGET_DIR="/data/home/jvnting/cnpo/models/MUSE/MUSE-News"
            MAX_LEN=2048
        elif [ "$CORPUS" = "books" ]; then
            # Books数据配置
            DATA_DIR="/data/home/jvnting/cnpo/data/books/raw"
            FORGET_DATA="$DATA_DIR/forget.txt"
            RETAIN_DATA="$DATA_DIR/retain1.txt"
            OUT_DIR_BASE="/data/home/jvnting/cnpo/ckpt/books/final"
            COEFF_TYPE='cosine'
            TARGET_DIR="/data/home/jvnting/cnpo/models/MUSE/MUSE-Books"
            MAX_LEN=2048
        else
            DATA_DIR="/data/home/jvnting/cnpo/data/pii/scal-${scal}/raw/synthetic_data.json"
            OUT_DIR_BASE="/data/home/jvnting/cnpo/ckpt/pii/final/scal/"
            COEFF_TYPE='semantic_similarity'
            # TARGET_DIR="/data/home/jvnting/cnpo/models/pii_7B_paraphrase-scal-${scal}"
            TARGET_DIR="/data/home/jvnting/cnpo/models/pii_7B_test_target-scal-${scal}"
            MAX_LEN=512
        fi

        LLAMA_DIR="/data/home/jvnting/cnpo/meta-llama"
        
        # 遍历算法
        for ALGO_TYPE in "${ALGO_TYPES[@]}"; do
            echo "========================================================="
            echo "=== Processing CORPUS: $CORPUS, ALGO_TYPE: $ALGO_TYPE ==="
            echo "========================================================="

            if [ "$ALGO_TYPE" = "ga" ]; then
                ALGOS=('ga' 'ga_gdr' 'ga_klr')
                beta=1 

                echo "Model Dir is: $TARGET_DIR"
                echo "Running GA algorithms..."
                for current_algo in "${ALGOS[@]}"; do
                    echo "Running Method: ${current_algo}..."
                    out_dir="$OUT_DIR_BASE/${current_algo}_beta=${beta}"
                    python -u baselines/unlearn.py \
                        --algo "$current_algo" \
                        --model_dir "$TARGET_DIR" --tokenizer_dir "$LLAMA_DIR" \
                        --data_file "$FORGET_DATA" \
                        --retain_data_file "$RETAIN_DATA" \
                        --out_dir "$out_dir" \
                        --max_len "$MAX_LEN" --epochs "$EPOCHS" --lr "$LR" \
                        --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
                        --corpus "$CORPUS" \
                        --beta "$beta"
                done

            elif [ "$ALGO_TYPE" = "npo" ]; then
                ALGOS=('npo' 'npo_gdr' 'npo_klr')
                # ALGOS=('npo_klr')
                beta=0.1
                echo "========================================================="
                echo "Running Method: ${current_algo}...with beta=${beta} and lr=${LR}"
                echo "========================================================="
                if [ "$CORPUS" = "news" ] || [ "$CORPUS" = "books" ]; then
                    DATA_ARG="--data_file $FORGET_DATA"
                else
                    DATA_ARG="--data_file $DATA_DIR"
                    SCAL="--scal $scal"
                fi
                echo "Model Dir is: $TARGET_DIR"
                echo "Running NPO algorithms..."
                for current_algo in "${ALGOS[@]}"; do
                    out_dir="$OUT_DIR_BASE/${current_algo}-beta=${beta}"
                    
                    echo "=== Running $current_algo algorithm ==="
                    CMD="python -u baselines/unlearn.py \
                        --algo \"$current_algo\" \
                        --model_dir \"$TARGET_DIR\" \
                        --tokenizer_dir \"$LLAMA_DIR\" \
                        $DATA_ARG \
                        --out_dir \"$out_dir\" \
                        --max_len \"$MAX_LEN\" \
                        --epochs \"$EPOCHS\" \
                        --lr \"$LR\" \
                        --per_device_batch_size \"$PER_DEVICE_BATCH_SIZE\" \
                        --corpus \"$CORPUS\" \
                        --beta \"$beta\" \
                        $SCAL"
                    if [ "$current_algo" = "npo_klr" ] || [ "$current_algo" = "npo_gdr" ]; then
                        CMD="$CMD --retain_data_file \"$RETAIN_DATA\""
                        echo "Adding retain data for KLR/GDR variant..."
                    fi
                    
                    eval $CMD
                    echo "Finished $current_algo. Results saved to $out_dir"
                done

            elif [ "$ALGO_TYPE" = "simnpo" ]; then
                ALGOS=('simnpo' 'simnpo_gdr' 'simnpo_klr')
                # ALGOS=('simnpo_klr')
                beta=0.5
                if [ "$CORPUS" = "news" ] || [ "$CORPUS" = "books" ]; then
                    DATA_ARG="--data_file $FORGET_DATA --retain_data_file $RETAIN_DATA "
                else
                    DATA_ARG="--data_file $DATA_DIR"
                    SCAL="--scal $scal"
                fi
                echo "Model Dir is: $TARGET_DIR"
                echo "Running SimNPO algorithms..."
                for current_algo in "${ALGOS[@]}"; do
                    echo "========================================================="
                    echo "Running Method: ${current_algo}...with beta=${beta} and lr=${LR}"
                    echo "========================================================="
                    out_dir="$OUT_DIR_BASE/${current_algo}_beta=${beta}"
                    python -u baselines/unlearn.py \
                        --algo "$current_algo" \
                        --model_dir "$TARGET_DIR" --tokenizer_dir "$LLAMA_DIR" \
                        $DATA_ARG \
                        --out_dir "$out_dir" \
                        --max_len "$MAX_LEN" --epochs "$EPOCHS" --lr "$LR" \
                        --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
                        --version $VERSION \
                        --corpus "$CORPUS" \
                        --beta "$beta" \
                        $SCAL
                done

            elif [ "$ALGO_TYPE" = "cnpo" ]; then
                for neg_sample in "${NEG_SAMPLE_NUMS[@]}"; do
                    #'cont_npo_gdr' 'cont_npo_klr'
                    ALGOS=('cont_npo_gdr')
                    # ALGOS=('cont_npo_klr')
                    echo "========================================================="
                    echo "========= Processing NEG_SAMPLE_NUM: $neg_sample ========"
                    echo "========= Model Dir is: $TARGET_DIR======================"
                    echo "========= Running CNPO algorithms...====================="
                    echo "========================================================="
                    # beta_values=($(seq 0.11 0.01 0.15))
                    beta_values=(0.1)
                    if [ "$CORPUS" = "news" ] || [ "$CORPUS" = "books" ]; then
                        DATA_ARG="--data_file $FORGET_DATA --retain_data_file $RETAIN_DATA"
                        out_dir="$OUT_DIR_BASE/${current_algo}_beta=${beta}_coeff-type=${COEFF_TYPE}_neg-samples=${neg_sample}"
                    else
                        DATA_ARG="--data_file $DATA_DIR"
                        SCAL="--scal $scal"
                        out_dir="$OUT_DIR_BASE/${current_algo}_beta=${beta}_coeff-type=${COEFF_TYPE}_neg-samples=${neg_sample}_scal_${scal}"
                    fi

                    for current_algo in "${ALGOS[@]}"; do
                        for beta in "${beta_values[@]}"; do
                            echo "========================================================="
                            echo "Running Method: ${current_algo}...with negative sample num=${neg_sample}, beta=${beta} and coeff-type=${COEFF_TYPE}, lr=${LR}"
                            echo "========================================================="
                            # out_dir="$OUT_DIR_BASE/${current_algo}_beta=${beta}_coeff-type=${COEFF_TYPE}_neg-samples=${neg_sample}"
                            
                            python -u baselines/unlearn.py \
                                --algo "$current_algo" \
                                --model_dir "$TARGET_DIR" --tokenizer_dir "$LLAMA_DIR" \
                                $DATA_ARG \
                                --out_dir "$out_dir" \
                                --max_len "$MAX_LEN" --epochs "$EPOCHS" --lr "$LR" \
                                --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
                                --coeff_type "$COEFF_TYPE" \
                                --neg_sample_num "$neg_sample" \
                                --version $VERSION \
                                --corpus "$CORPUS" \
                                --beta "$beta" \
                                --scal "$scal"
                        done
                    done
                done
            elif [ "$ALGO_TYPE" = "tv" ]; then
                if [ "$CORPUS" = "news" ] || [ "$CORPUS" = "books" ]; then
                    DATA_ARG="--data_file $FORGET_DATA"
                else
                    DATA_ARG="--data_file $DATA_DIR"
                    SCAL="--scal $scal"
                fi
                current_algo=$ALGO_TYPE
                echo "Model Dir is: $TARGET_DIR"
                echo "Running Task Vector algorithms..."
                echo "========================================================="
                echo "Running Method: ${current_algo}... and lr=${LR}"
                echo "========================================================="
                out_dir="$OUT_DIR_BASE/${current_algo}"
                python -u baselines/unlearn.py \
                    --algo "$current_algo" \
                    --model_dir "$TARGET_DIR" --tokenizer_dir "$LLAMA_DIR" \
                    $DATA_ARG \
                    --out_dir "$out_dir" \
                    --max_len "$MAX_LEN" --epochs "$EPOCHS" --lr "$LR" \
                    --per_device_batch_size "$PER_DEVICE_BATCH_SIZE" \
                    --version $VERSION \
                    --corpus "$CORPUS" \
                    $SCAL
            fi
        done
    done
done