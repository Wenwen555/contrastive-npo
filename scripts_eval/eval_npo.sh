#!/bin/bash
export CUDA_VISIBLE_DEVICES=3

# 配置参数
CORPORA=("pii")  # 评估数据集
ALGO_TYPES=("cnpo")  # 遗忘算法类型
SCALS=('10' '20' '40')  # pii数据集规模

# 基础目录设置
LLAMA_DIR="/data/home/jvnting/cnpo/meta-llama"  # 原始Llama模型目录
BASE_CKPT_DIR="/data/home/jvnting/cnpo/ckpt"    # 遗忘模型根目录

# 遍历所有数据集和算法
for CORPUS in "${CORPORA[@]}"; do
    for ALGO_TYPE in "${ALGO_TYPES[@]}"; do
        for SCAL in "${SCALS[@]}"; do
            echo "=== Evaluating CORPUS: $CORPUS, ALGO_TYPE: $ALGO_TYPE ==="
            
            # 根据算法类型设置参数
            case "$ALGO_TYPE" in
                "ga")
                    ALGOS=('ga' 'ga_gdr' 'ga_klr')
                    BETA=1
                    LOSS_TYPE="ga-beta-$BETA"
                    ;;
                "npo")
                    ALGOS=('npo' 'npo_gdr' 'npo_klr')
                    BETA=0.1
                    LOSS_TYPE="npo-beta-$BETA"
                    ;;
                "simnpo")
                    ALGOS=('simnpo' 'simnpo_gdr' 'simnpo_klr')
                    BETA=0.5
                    LOSS_TYPE="simnpo-beta-$BETA"
                    ;;
                "cnpo")
                    # 
                    ALGOS=('cont_npo_gdr')
                    BETA=(0.1)
                    # news and books 用cosine，而pii用semantic entropy
                    COEFF_TYPE="semantic_similarity"
                    NEG_SAMPLE_NUMS=(2 3 4)
                    LOSS_TYPE="cnpo-beta-${BETA}"
                    ;;
                "tv")
                    ALGOS=('tv')
                    # news and books 用cosine，而pii用semantic entropy
                    LOSS_TYPE="tv"
                    ;;
            esac

            if [ "$ALGO_TYPE" = "cnpo" ]; then
                for NEG_SAMPLE_NUM in "${NEG_SAMPLE_NUMS[@]}"; do
                    for beta in "${BETA[@]}"; do
                        for CURRENT_ALGO in "${ALGOS[@]}"; do
                            # 构建模型目录路径
                            BASE_MODEL_DIR="${BASE_CKPT_DIR}/${CORPUS}/final/scal/${CURRENT_ALGO}_beta=${BETA}_coeff-type=${COEFF_TYPE}_neg-samples=${NEG_SAMPLE_NUM}_scal_${SCAL}/"
                            
                            # Temporal address
                            # BASE_MODEL_DIR="${BASE_CKPT_DIR}/${CORPUS}/final/${CURRENT_ALGO}/"
                            # if [ -d "$BASE_MODEL_DIR" ]; then
                            #     echo "找到模型目录: $BASE_MODEL_DIR"
                            #     MODEL_DIR=$BASE_MODEL_DIR
                            #     CKPT_NAME=$(basename "$MODEL_DIR")
                                
                            #     echo "Evaluating model:"
                            #     echo "  Model dir: $MODEL_DIR"
                            #     echo "  Corpus: $CORPUS"
                            #     echo "  Algorithm: $CURRENT_ALGO"
                            #     echo "  Checkpoint: $CKPT_NAME"
                                

                            #     # 构建输出文件路径
                            #     if [ "$CORPUS" = "pii" ]; then
                            #         # OUT_FILE="output/${CORPUS}/final/pii_paraphrase_scal-${SCAL}/${LOSS_TYPE}/${CURRENT_ALGO}_${NEG_SAMPLE_NUM}n/${CKPT_NAME}.csv"
                            #         OUT_FILE="output/${CORPUS}/final/pii_test_target_scal-${SCAL}/${LOSS_TYPE}/${CURRENT_ALGO}_${NEG_SAMPLE_NUM}n/${CKPT_NAME}.csv"
                            #     else
                            #         OUT_FILE="output/${CORPUS}/final/${LOSS_TYPE}/${CURRENT_ALGO}_${NEG_SAMPLE_NUM}n/${CKPT_NAME}.csv"
                            #     fi
                            #     echo "  Saving to: ${OUT_FILE}"

                            #     # 检查是否已经有Output file了
                            #     if [ -f "$OUT_FILE" ]; then
                            #         echo "${CKPT_NAME}已经存在！跳过评估"
                            #     else
                            #         # 创建输出目录
                            #         mkdir -p "$(dirname "$OUT_FILE")"

                            #         # 执行评估
                            #         echo "${CURRENT_ALGO}_${NEG_SAMPLE_NUM}n"
                            #         python eval.py \
                            #             --model_dirs "$MODEL_DIR" \
                            #             --names "$CKPT_NAME" \
                            #             --corpus "$CORPUS" \
                            #             --out_file "$OUT_FILE" \
                            #             --loss_type "$LOSS_TYPE" \
                            #             --loss "${CURRENT_ALGO}_${NEG_SAMPLE_NUM}n"
                                        
                            #         echo "----------------------------------------"
                            #     fi
                            # else
                            #     echo "wrong !"
                            # fi
                            # 查找所有匹配的模型目录
                            for MODEL_DIR in ${BASE_MODEL_DIR}*; do
                                if [ "$MODEL_DIR" = "$BASE_MODEL_DIR" ]; then
                                    continue
                                fi
                                # eval checkpoint子目录
                                if [ -d "${MODEL_DIR}" ]; then
                                    echo "存在: $MODEL_DIR"
                                    
                                    # 提取checkpoint名称（最后一级目录名）
                                    CKPT_NAME=$(basename "$MODEL_DIR")
                                
                                    echo "Evaluating model:"
                                    echo "  Model dir: $MODEL_DIR"
                                    echo "  Corpus: $CORPUS"
                                    echo "  Algorithm: $CURRENT_ALGO"
                                    echo "  Checkpoint: $CKPT_NAME"

                                    # 构建输出文件路径
                                    if [ "$CORPUS" = "pii" ]; then
                                        # OUT_FILE="output/${CORPUS}/final/pii_paraphrase_scal-${SCAL}/${LOSS_TYPE}/${CURRENT_ALGO}_${NEG_SAMPLE_NUM}n/${CKPT_NAME}.csv"
                                        OUT_FILE="output/${CORPUS}/final/scalability/pii_scal-${SCAL}/${LOSS_TYPE}/${CURRENT_ALGO}_${NEG_SAMPLE_NUM}n/${CKPT_NAME}.csv"
                                    else
                                        OUT_FILE="output/${CORPUS}/final/ablation/${LOSS_TYPE}/${CURRENT_ALGO}_${NEG_SAMPLE_NUM}n/${CKPT_NAME}.csv"
                                    fi
                                    echo "  Saving to: ${OUT_FILE}"
                                    # 检查是否已经有Output file了
                                    if [ -f "$OUT_FILE" ]; then
                                        echo "${CKPT_NAME}已经存在！跳过评估"
                                    else
                                        # 创建输出目录
                                        mkdir -p "$(dirname "$OUT_FILE")"

                                        echo "${CURRENT_ALGO}_${NEG_SAMPLE_NUM}n"
                                        # 执行评估
                                        python eval.py \
                                            --model_dirs "$MODEL_DIR" \
                                            --names "$CKPT_NAME" \
                                            --corpus "$CORPUS" \
                                            --out_file "$OUT_FILE" \
                                            --loss_type "$LOSS_TYPE" \
                                            --loss "${CURRENT_ALGO}_${NEG_SAMPLE_NUM}n" \
                                            --scal "$SCAL"
                                        
                                        echo "----------------------------------------"
                                    fi
                                else
                                    echo "不存在: $MODEL_DIR"
                                fi
                            done
                        done
                    done
                done
            else 
                for CURRENT_ALGO in "${ALGOS[@]}"; do
                    # 构建模型目录路径
                    if [ $CURRENT_ALGO = "tv" ]; then 
                        BASE_MODEL_DIR="${BASE_CKPT_DIR}/${CORPUS}/final/${CURRENT_ALGO}/"
                    else
                        BASE_MODEL_DIR="${BASE_CKPT_DIR}/${CORPUS}/final/${CURRENT_ALGO}_beta=${BETA}/"
                    fi
                    if [ -d "$BASE_MODEL_DIR" ]; then
                        echo "找到模型目录: $BASE_MODEL_DIR"
                        MODEL_DIR=$BASE_MODEL_DIR
                        CKPT_NAME=$(basename "$MODEL_DIR")
                        
                        echo "Evaluating model:"
                        echo "  Model dir: $MODEL_DIR"
                        echo "  Corpus: $CORPUS"
                        echo "  Algorithm: $CURRENT_ALGO"
                        echo "  Checkpoint: $CKPT_NAME"

                        # 构建输出文件路径
                        OUT_FILE="output/${CORPUS}/final/${LOSS_TYPE}.csv"
                        echo "  Saving to: ${OUT_FILE}"
                        # 检查是否已经有Output file了
                        if [ -f "$OUT_FILE" ]; then
                            echo "${CKPT_NAME}已经存在！跳过评估"
                        else
                            # 创建输出目录
                            mkdir -p "$(dirname "$OUT_FILE")"

                            # 执行评估
                            python eval.py \
                                --model_dirs "$MODEL_DIR" \
                                --names "$CKPT_NAME" \
                                --corpus "$CORPUS" \
                                --out_file "$OUT_FILE" \
                                --loss_type "$LOSS_TYPE" \
                                --loss "${CURRENT_ALGO}"
                                
                            
                            echo "----------------------------------------"
                        fi
                    else
                        echo "wrong !"
                    fi
                    # 查找所有匹配的模型目录
                    for MODEL_DIR in ${BASE_MODEL_DIR}*; do
                        if [ "$MODEL_DIR" = "$BASE_MODEL_DIR" ]; then
                            continue
                        fi
                        # eval checkpoint子目录
                        if [ -d "${MODEL_DIR}" ]; then
                            echo "存在: $MODEL_DIR"
                            
                            # 提取checkpoint名称（最后一级目录名）
                            CKPT_NAME=$(basename "$MODEL_DIR")
                        
                            echo "Evaluating model:"
                            echo "  Model dir: $MODEL_DIR"
                            echo "  Corpus: $CORPUS"
                            echo "  Algorithm: $CURRENT_ALGO"
                            echo "  Checkpoint: $CKPT_NAME"

                            # 构建输出文件路径
                            if [ "$CORPUS" = "pii" ]; then
                                OUT_FILE="output/${CORPUS}/final/pii_paraphrase_scal-${SCAL}/${LOSS_TYPE}/${CURRENT_ALGO}/${CKPT_NAME}.csv"
                            else
                                OUT_FILE="output/${CORPUS}/final/${LOSS_TYPE}/${CURRENT_ALGO}/${CKPT_NAME}.csv"
                            fi
                            echo "  Saving to: ${OUT_FILE}"
                            # 检查是否已经有Output file了
                            if [ -f "$OUT_FILE" ]; then
                                echo "${CKPT_NAME}已经存在！跳过评估"
                            else
                                # 创建输出目录
                                mkdir -p "$(dirname "$OUT_FILE")"
                                # 执行评估
                                python eval.py \
                                    --model_dirs "$MODEL_DIR" \
                                    --names "$CKPT_NAME" \
                                    --corpus "$CORPUS" \
                                    --out_file "$OUT_FILE" \
                                    --loss_type "$LOSS_TYPE" \
                                    --loss "${CURRENT_ALGO}"
                                
                                echo "----------------------------------------"
                            fi
                        else
                            echo "不存在: $MODEL_DIR"
                        fi
                    done
                done
            fi
        done
    done
done