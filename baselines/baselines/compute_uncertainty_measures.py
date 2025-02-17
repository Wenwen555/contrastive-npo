import logging
import numpy as np
import torch

from semantic_entropy import get_semantic_ids
from semantic_entropy import logsumexp_by_id
from semantic_entropy import predictive_entropy_rao
from semantic_entropy import EntailmentPythia

class SemanticEntropyCalculator:
    def __init__(self, strict_entailment=False):
        """
        初始化语义熵计算器
        Args:
            strict_entailment: 是否使用严格的语义蕴含判断
        """
        logging.info('Loading entailment model...')
        self.entailment_model = EntailmentPythia(None,None,
                                                 local_model_path='/mnt/wenjt5/muse/model/pythia/pythia-410m-news')
        self.strict_entailment = strict_entailment
        self.model = self.entailment_model.model
        self.tokenizer = self.entailment_model.tokenizer  # 假设EntailmentPythia包含tokenizer

    def ids_to_text(self, input_ids, attention_mask=None):
        """
        将input_ids解码为文本
        Args:
            input_ids: torch.Tensor 形状 [batch_size, seq_len]
            attention_mask: torch.Tensor 形状 [batch_size, seq_len]
        Returns:
            list: 解码后的文本列表
        """
        texts = []
        for ids, mask in zip(input_ids, attention_mask or [None]*len(input_ids)):
            if mask is not None:
                valid_length = torch.sum(mask).item()
                ids = ids[:valid_length]
            text = self.tokenizer.decode(
                ids.cpu().numpy(),
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            texts.append(text.strip())
        return texts

    def calculate_entropy(self, input_ids, log_liks, attention_mask=None, num_generations=10):
        """
        计算批量数据的语义熵
        Args:
            input_ids: torch.Tensor 形状 [batch_size, seq_len]
            log_liks: list 每个元素的形状 [seq_len]，对应每个序列的log likelihood
            attention_mask: torch.Tensor 形状 [batch_size, seq_len]
            num_generations: 每个样本生成的响应数量
        Returns:
            list: 每个样本的预测熵值
        """
        entropies = []
        
        # 将tensor转换为文本
        responses = self.ids_to_text(input_ids, attention_mask)
        
        # 计算语义ID
        semantic_ids = get_semantic_ids(
            responses, 
            model=self.model,
            strict_entailment=self.strict_entailment,
            example=None  # 根据实际情况调整
        )
        
        # 处理log likelihoods
        processed_log_liks = [np.mean(log_lik) for log_lik in log_liks]
        
        # 计算语义熵
        log_likelihood_per_semantic_id = logsumexp_by_id(
            semantic_ids, 
            processed_log_liks,
            agg='sum_normalized'
        )
        pe_rao = predictive_entropy_rao(log_likelihood_per_semantic_id)
        entropies.append(pe_rao)

        return entropies

# 使用示例
if __name__ == '__main__':
    # 初始化计算器
    calculator = SemanticEntropyCalculator(strict_entailment=True)
    
    # 示例输入 (假设batch_size=2, seq_len=10)
    input_ids = torch.randint(0, 1000, (2, 10))
    attention_mask = torch.ones_like(input_ids)
    log_liks = [np.random.rand(10) for _ in range(2)]  # 假设每个序列的log likelihood
    
    # 计算熵
    entropies = calculator.calculate_entropy(
        input_ids=input_ids,
        log_liks=log_liks,
        attention_mask=attention_mask,
        num_generations=10
    )
    
    print("预测熵值:", entropies)
