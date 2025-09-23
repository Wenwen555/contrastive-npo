from llm2vec import LLM2Vec
import os
import json
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
from pathlib import Path
from tqdm import tqdm


def load_and_compute_similarity(retain_path, forget_path, model, tokenizer):
    """
    Load retain and forget datasets from JSON files, compute embeddings, and calculate cosine similarity.
    
    Args:
        retain_path (str): Path to the retain dataset JSON file.
        forget_path (str): Path to the forget dataset JSON file.
        model_name (str): Name of the SentenceTransformer model to use (default: 'all-MiniLM-L6-v2').
    
    Returns:
        tuple: (retain_data_with_embeds, forget_data_with_embeds, similarity_matrix)
            - retain_data_with_embeds: List of retain data with embedded "question" stored under "embed".
            - forget_data_with_embeds: List of forget data with embedded "question" stored under "embed".
            - similarity_matrix: Tensor of cosine similarities between retain and forget questions.
    """

    # Wrapper for encoding and pooling operations
    l2v = LLM2Vec(model, tokenizer, pooling_mode="mean", max_length=512)
    
    # Load datasets from JSON files
    with open(retain_path, 'r') as f:
        retain_data = json.load(f)
    with open(forget_path, 'r') as f:
        forget_data = json.load(f)
    
    # Extract questions from both datasets
    # retain_concat_list = [f"Question: {item['question']} Answer: {item['answer']}" for item in retain_data]
    # forget_concat_list = [f"Question: {item['question']} Answer: {item['answer']}" for item in forget_data]
    
    retain_queries = [
        [item]
        for item in retain_data
    ]
    forget_queries = [
        [item]
        for item in forget_data
    ]

    # Initialize similarity matrix
    similarity_matrix = torch.zeros(len(retain_data), len(forget_data))

    for i, retain_sample in tqdm(enumerate(retain_queries), total=len(retain_queries), desc="Processing Retain Samples"):
        # Encode current retain sample
        q_rep = l2v.encode(retain_sample, convert_to_tensor=True)  # shape: (embed_dim,)
        q_rep_norm = torch.nn.functional.normalize(q_rep, p=2, dim=1)  # L2 normalize   
        # Store embedding in original data
        # retain_data[i]["embed"] = q_rep_norm.tolist()
        
        for j, forget_sample in tqdm(enumerate(forget_queries), total=len(forget_queries), desc=f"Matching Retain {i+1}/{len(retain_queries)}", leave=False):
            # Encode current forget sample
            d_rep = l2v.encode(forget_sample, convert_to_tensor=True)  # shape: (embed_dim,)
            d_rep_norm = torch.nn.functional.normalize(d_rep, p=2, dim=1)
            # Compute cosine similarity with torch.dot
            similarity_matrix[i, j] = torch.trace(torch.mm(q_rep_norm,d_rep_norm.transpose(0,1)))
            # torch.dist(sentence_embedding_norm_q, sentence_embedding_norm_d)
            # torch.nn.functional.cosine_similarity(sentence_embedding_norm_q.unsqueeze(0), sentence_embedding_norm_d.unsqueeze(0)
            # if i == 0:
            #     forget_data[j]["embed"] = d_rep_norm.tolist()
    
    return retain_data, forget_data, similarity_matrix



def process_paired_files(base_path, corpus, model, tokenizer):
    # 确保输出目录存在
    output_dir = Path(f"/data/home/jvnting/cnpo/data/{corpus}/smiliarity")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 构建文件路径
    retain_path = os.path.join(base_path, "retain1.json")
    forget_path = os.path.join(base_path, "forget.json")
    
    # 检查文件是否存在
    if not (os.path.exists(retain_path) and os.path.exists(forget_path)):
        print(f"Warning: Missing files, skipping")
    
    # 计算相似度矩阵
    _, _, similarity_matrix = load_and_compute_similarity(
        retain_path, forget_path, model, tokenizer
    )
    
    print(f"Cosine similarity matrix shape: {similarity_matrix.shape}")
    
    # 保存结果
    output_filename = f"similarity-{corpus}.pt"
    output_path = output_dir / output_filename
    torch.save(similarity_matrix, output_path)
    print(f"Saved similarity matrix to {output_path}")


# Example usage:
if __name__ == "__main__":
    # Loading base Mistral model, along with custom code that enables bidirectional connections in decoder-only LLMs. MNTP LoRA weights are merged into the base model.
    tokenizer = AutoTokenizer.from_pretrained(
        "/data/home/jvnting/Unlearn-Simple/TOFU/paper_models/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
    )
    tokenizer.pad_token = tokenizer.eos_token

    config = AutoConfig.from_pretrained(
        "/data/home/jvnting/Unlearn-Simple/TOFU/paper_models/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        trust_remote_code=True
    )

    model = AutoModel.from_pretrained(
        "/data/home/jvnting/Unlearn-Simple/TOFU/paper_models/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        trust_remote_code=True,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )

    model = PeftModel.from_pretrained(
        model,
        "/data/home/jvnting/Unlearn-Simple/TOFU/paper_models/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
    )
    model = model.merge_and_unload()  # This can take several minutes on cpu
    # Loading unsupervised SimCSE model. This loads the trained LoRA weights on top of MNTP model. Hence the final weights are -- Base model + MNTP (LoRA) + SimCSE (LoRA).
    model = PeftModel.from_pretrained(
        model, "/data/home/jvnting/Unlearn-Simple/TOFU/paper_models/mistral-7b-instruct-v2-mntp-sup"
    )

    corpus = 'news'
    base_path = f"/data/home/jvnting/cnpo/data/{corpus}/raw"
    process_paired_files(base_path, corpus, model, tokenizer)

    # print("Example retain entry with embed:", retain_data[0])
    # print("Example forget entry with embed:", forget_data[0]) 
    # with open("/data/home/jvnting/Unlearn-Simple/TOFU/TOFU_data_embed/retain95.json", 'w') as f:
    #     json.dump(retain_data, f)
    # with open("/data/home/jvnting/Unlearn-Simple/TOFU/TOFU_data_embed/forget05.json", 'w') as f:
    #     json.dump(forget_data, f)
