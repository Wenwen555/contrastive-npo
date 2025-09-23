
ALGO_TYPES=("cnpo")
scals=(10)
NEG_SAMPLE_NUMS=(3 4)

for ALGO_TYPE in "${ALGO_TYPES[@]}"; do
    if [ "$ALGO_TYPE" = "cnpo" ]; then
        for scal in "${scals[@]}"; do
            for NEG_SAMPLE_NUM in "${NEG_SAMPLE_NUMS[@]}"; do
                BASE_MODEL_DIR="/data/home/jvnting/cnpo/temp/pii/final/scalability/scal-${scal}/cnpo-beta-0.1/cont_npo_gdr_${NEG_SAMPLE_NUM}n/"
                # BASE_MODEL_DIR="/data/home/jvnting/cnpo/temp/pii/scal-5/cnpo-beta-0.5/"
                CHECKPOINTS=$(ls -dv ${BASE_MODEL_DIR}*)
                # echo "Eval Checkpoint path: $CHECKPOINTS"
                Eval_saved_path="/data/home/jvnting/cnpo/output/pii/utility/scalability/scal-${scal}/"
                for MODEL_DIR in $CHECKPOINTS; do
                    name=$(basename "$MODEL_DIR")
                    if [ "$name" != "checkpoint-198" ]; then
                        echo "跳过非目标checkpoint: $name"
                        continue
                    fi
                    eval_result_dir="${Eval_saved_path}/cnpo-beta-0.1-${NEG_SAMPLE_NUM}n/${ALGO_TYPE}_${NEG_SAMPLE_NUM}n_${name}"
                    if [ -f "$eval_result_dir" ]; then
                        echo "${ALGO_TYPE}_${NEG_SAMPLE_NUM}n_${name}已经存在！跳过评估"
                    else
                        echo "存在: $MODEL_DIR"
                        echo "实验规模为：$scal"
                        eval_path="${MODEL_DIR}/verbmem_f/log.json"
                        echo "Save to path: ${eval_result_dir}"
                        python -u metrics/metric-pii/fluency.py \
                            --eval_path $eval_path \
                            --name "${ALGO_TYPE}_${NEG_SAMPLE_NUM}n_${name}" \
                            --scal $scal
                    fi
                done
            done
        done    
    else
        if [ "$ALGO_TYPE" = "simnpo" ]; then
            BASE_MODEL_DIR="/data/home/jvnting/cnpo/temp/pii/final/scal-5/simnpo-beta-0.5/simnpo/"
        elif [ "$ALGO_TYPE" = "npo" ]; then
            BASE_MODEL_DIR="/data/home/jvnting/cnpo/temp/pii/final/scal-5/npo-beta-0.1/npo/"
        fi
        # CHECKPOINTS=$(ls -dv ${BASE_MODEL_DIR}* | head -n 2)
        CHECKPOINTS=$(ls -dv ${BASE_MODEL_DIR}* | tail -n 2)
        echo "Eval Checkpoint path: $CHECKPOINTS"
        for MODEL_DIR in $CHECKPOINTS; do
            echo "存在: $MODEL_DIR"
            name=$(basename "$MODEL_DIR")
            eval_path="${MODEL_DIR}/verbmem_f/log.json"
            echo "Save to name: ${ALGO_TYPE}_${name}_beta=0.1"
            python -u metrics/metric-pii/fluency.py \
                --eval_path $eval_path \
                --name "${ALGO_TYPE}_${name}_beta=0.1"
        done
    fi
done

# python -u metrics/metric-pii/fluency.py \
#     --eval_path "/data/home/jvnting/cnpo/temp/pii/scal-5/pretrained_on_pii/verbmem_f/log.json" \
#     --name "pretrain_model"