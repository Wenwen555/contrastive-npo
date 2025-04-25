export CUDA_VISIBLE_DEVICES=1
# 定义模型目录和名称的遍历列表
model_dirs=(
        # "/data/home/jvnting/cnpo/ckpt/pii/cont_npo_2n_cosine_7B_3/checkpoint-50/"
        # "/data/home/jvnting/cnpo/ckpt/pii/cont_npo_2n_cosine_7B_2/checkpoint-50/"
        "/data/home/jvnting/cnpo/models/pii_7B/checkpoint-500/"
        # "/data/home/jvnting/cnpo/ckpt/pii/npo_2n_cosine_7B_2/"
)

# 此处处理原本的llama，是我担心lora的性能会影响Unlearning过程！
names=(
    # "cnpo_2n_cosine_v3_epoch1"
    "pii-7B"
)

# 固定参数
coeff_type="cosine"
neg_sample_num=2
corpus="pii"
scal=5

# 双重循环遍历所有组合
for ((i=0; i<${#model_dirs[@]}; i++)); do
    model_dir="${model_dirs[i]}"
    name="${names[i]}"
    
    echo "Running evaluation with:"
    echo "  Model dir: $model_dir"
    echo "  Name: $name"
    echo "  Coeff type: $coeff_type"
    echo "  Neg sample num: $neg_sample_num"
    
    python eval.py \
        --model_dirs "$model_dir" \
        --basemodel_dir "/data/home/jvnting/cnpo/meta-llama/" \
        --names "$name" \
        --corpus "$corpus" \
        --out_file "output/pii/${name}.csv" \
        --scal "$scal"
    
    echo "----------------------------------------"
done


# base_dir='/root/cnpo/ckpt/news/cont_npo_2n_cosine_7B'
# for checkpoint_path in "$base_dir"/*; do
#     echo $checkpoint_path
#     dir_name=$(basename "$checkpoint_path")
#     if [ -d "$checkpoint_path" ]; then
#         echo "Processing check point: ${dir_name}"
#         python eval.py \
#             --model_dirs $checkpoint_path \
#             --names "7b-cnpo-v3-cos-2n-unlearned--${dir_name}" \
#             --corpus news \
#             --out_file output/news/"7b-cnpo-v3-cos-2nunlearned-${dir_name}.csv"
#     else
#         echo "Warning: This is not folder!"
#     fi
# done
