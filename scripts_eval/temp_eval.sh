# 执行评估
export CUDA_VISIBLE_DEVICES=1
# 配置参数
CORPUS=("books")  # 评估数据集
scal=5
model_dir="/data/home/jvnting/cnpo/models/MUSE/MUSE-Books/"  
CKPT_NAME="original-model"
OUT_FILE="output/${CORPUS}/final/${CKPT_NAME}.csv"
echo "Evaluating form ${model_dir}"
echo "Output to ${OUT_FILE}"
python eval.py \
    --model_dirs "$model_dir" \
    --names "$CKPT_NAME" \
    --corpus "$CORPUS" \
    --out_file "$OUT_FILE" \
    --loss_type "$CKPT_NAME" \
    --loss "$CKPT_NAME"
echo "----------------------------------------"