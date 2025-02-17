"""Implement semantic entropy."""
import os
import pickle
import logging

import numpy as np

import wandb

import torch
import torch.nn.functional as F

from transformers import AutoModelForSequenceClassification, AutoTokenizer

from baselines.baselines.utils import load_model_and_tokenizer

from baselines.baselines.utils import save
from baselines.baselines.utils import md5hash


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class BaseEntailment:
    def save_prediction_cache(self):
        pass


class EntailmentDeberta(BaseEntailment):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli").to(DEVICE)

    def check_implication(self, text1, text2, *args, **kwargs):
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(DEVICE)
        # The model checks if text1 -> text2, i.e. if text2 follows from text1.
        # check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        largest_index = torch.argmax(F.softmax(logits, dim=1))  # pylint: disable=no-member
        prediction = largest_index.cpu().item()
        if os.environ.get('DEBERTA_FULL_LOG', False):
            logging.info('Deberta Input: %s -> %s', text1, text2)
            logging.info('Deberta Prediction: %s', prediction)

        return prediction


class EntailmentLLM(BaseEntailment):

    entailment_file = 'entailment_cache.pkl'

    def __init__(self, entailment_cache_id, entailment_cache_only):
        self.prediction_cache = self.init_prediction_cache(entailment_cache_id)
        self.entailment_cache_only = entailment_cache_only
        # 将方法添加到EntailmentLLM类


    def init_prediction_cache(self, entailment_cache_id):
        if entailment_cache_id is None:
            return dict()

        logging.info('Restoring prediction cache from %s', entailment_cache_id)

        api = wandb.Api()
        run = api.run(entailment_cache_id)
        run.file(self.entailment_file).download(
            replace=True, exist_ok=False, root=wandb.run.dir)

        with open(f'{wandb.run.dir}/{self.entailment_file}', "rb") as infile:
            return pickle.load(infile)

    def save_prediction_cache(self):
        # Write the dictionary to a pickle file.
        save(self.prediction_cache, self.entailment_file)

    def check_implication(self, text1, text2, example=None):
        if example is None:
            raise ValueError
        prompt = self.equivalence_prompt(text1, text2, example['question'])

        logging.info('%s input: %s', self.name, prompt)

        hashed = md5hash(prompt)
        if hashed in self.prediction_cache:
            logging.info('Restoring hashed instead of predicting with model.')
            response = self.prediction_cache[hashed]
        else:
            if self.entailment_cache_only:
                raise ValueError
            response = self.predict(prompt, temperature=0.02)
            self.prediction_cache[hashed] = response

        logging.info('%s prediction: %s', self.name, response)

        binary_response = response.lower()[:30]
        if 'entailment' in binary_response:
            return 2
        elif 'neutral' in binary_response:
            return 1
        elif 'contradiction' in binary_response:
            return 0
        else:
            logging.warning('MANUAL NEUTRAL!')
            return 1

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
        for ids, mask in zip(input_ids, attention_mask):
            if mask is not None:
                # 根据mask截断有效部分
                valid_length = torch.sum(mask).item()
                ids = ids[:valid_length]
            text = self.model.tokenizer.decode(
                ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            texts.append(text.strip())
        return texts

    def compute_semantic_entropy(self, input_ids, attention_mask=None, example=None):
        """
        基于input_ids计算语义熵的完整流程
        Args:
            input_ids: torch.Tensor 形状 [batch_size, seq_len]
            attention_mask: torch.Tensor 形状 [batch_size, seq_len]
            example: 上下文示例（用于entailment判断）
        Returns:
            dict: 包含三种熵值的字典
        """
        # 1. 解码为文本
        generated_texts = self.ids_to_text(input_ids, attention_mask)

        # 2. 获取语义ID
        semantic_ids = get_semantic_ids(
            generated_texts,
            model=self,
            example=example
        )

        # 3. 计算各类型熵值
        log_probs = []  # 需要根据实际获取对数概率
        cluster_entropy = cluster_assignment_entropy(semantic_ids)

        is_similar = (semantic_ids[0] == semantic_ids[1]).item() if len(input_ids) == 2 else None
        return {
            'cluster_entropy': cluster_entropy,
            'is_similar': is_similar,  # 直接判断结果
            'semantic_ids': semantic_ids,
            'generated_texts': generated_texts
        }



# class EntailmentGPT4(EntailmentLLM):
#
#     def __init__(self, entailment_cache_id, entailment_cache_only):
#         super().__init__(entailment_cache_id, entailment_cache_only)
#         self.name = 'gpt-4'
#
#     def equivalence_prompt(self, text1, text2, question):
#
#         prompt = f"""We are evaluating answers to the question \"{question}\"\n"""
#         prompt += "Here are two possible answers:\n"
#         prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
#         prompt += "Does Possible Answer 1 semantically entail Possible Answer 2? Respond with entailment, contradiction, or neutral."""
#
#         return prompt
#
#     def predict(self, prompt, temperature):
#         return oai.predict(prompt, temperature, model=self.name)
#
#
# class EntailmentGPT35(EntailmentGPT4):
#
#     def __init__(self, entailment_cache_id, entailment_cache_only):
#         super().__init__(entailment_cache_id, entailment_cache_only)
#         self.name = 'gpt-3.5'
#
#
# class EntailmentGPT4Turbo(EntailmentGPT4):
#
#     def __init__(self, entailment_cache_id, entailment_cache_only):
#         super().__init__(entailment_cache_id, entailment_cache_only)
#         self.name = 'gpt-4-turbo'


# class EntailmentLlama(EntailmentLLM):
#
#     def __init__(self, entailment_cache_id, entailment_cache_only, name):
#         super().__init__(entailment_cache_id, entailment_cache_only)
#         self.name = name
#         self.model = HuggingfaceModel(
#             name, stop_sequences='default', max_new_tokens=30)
#
#     def equivalence_prompt(self, text1, text2, question):
#
#         prompt = f"""We are evaluating answers to the question \"{question}\"\n"""
#         prompt += "Here are two possible answers:\n"
#         prompt += f"Possible Answer 1: {text1}\nPossible Answer 2: {text2}\n"
#         prompt += "Does Possible Answer 1 semantically entail Possible Answer 2? Respond only with entailment, contradiction, or neutral.\n"""
#         prompt += "Response:"""
#
#         return prompt
#
#     def predict(self, prompt, temperature):
#         predicted_answer, _, _ = self.model.predict(prompt, temperature)
#         return predicted_answer


class EntailmentPythia(EntailmentLLM):
    """Entailment module for locally stored Pythia models"""

    def __init__(self, entailment_cache_id, entailment_cache_only, local_model_path):
        """
        Args:
            local_model_path: 本地模型路径（包含config.json和模型文件的目录）
        """
        super().__init__(entailment_cache_id, entailment_cache_only)

        # 从路径中提取模型名称（可选）
        self.model_name = os.path.basename(local_model_path)
        self.name = f"pythia-{self.model_name}"

        # 初始化本地模型
        self.model, self.tokenizer = load_model_and_tokenizer(
            model_dir=local_model_path,  # 使用本地路径
            tokenizer_dir=local_model_path
        )

    def equivalence_prompt(self, text1, text2, question):
        """构造适用于Pythia的提示模板"""
        prompt = (
            "### Instruction:\n"
            f"Given the question: '{question}'\n"
            "Evaluate if Answer 1 logically implies Answer 2.\n\n"
            "### Input:\n"
            f"Answer 1: {text1}\n"
            f"Answer 2: {text2}\n\n"
            "### Response:\n"
            "Judgment (only one word):"
        )
        return prompt

    def predict(self, prompt, temperature):
        """带本地模型参数的生成配置"""
        predicted_answer, _, _ = self.model.predict(
            prompt,
            temperature=temperature,
            # top_p=0.90,  # 降低采样随机性
            # top_k=50,  # 限制候选token数量
            # repetition_penalty=1.15,  # 抑制重复
            # do_sample=True  # 确保启用采样
        )
        return predicted_answer.strip().lower()

def context_entails_response(context, responses, model):
    votes = []
    for response in responses:
        votes.append(model.check_implication(context, response))
    return 2 - np.mean(votes)

def get_semantic_ids(strings_list, model, strict_entailment=False, example=None):
    """Group list of predictions into semantic meaning."""

    def are_equivalent(text1, text2):

        implication_1 = model.check_implication(text1, text2, example=example)
        implication_2 = model.check_implication(text2, text1, example=example)  # pylint: disable=arguments-out-of-order
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

        if strict_entailment:
            semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)

        else:
            implications = [implication_1, implication_2]
            # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
            semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

        return semantically_equivalent

    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id.
    next_id = 0
    for i, string1 in enumerate(strings_list):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i+1, len(strings_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids

    return semantic_set_ids

def logsumexp_by_id(semantic_ids, log_likelihoods, agg='sum_normalized'):
    """Sum probabilities with the same semantic id.

    Log-Sum-Exp because input and output probabilities in log space.
    """
    unique_ids = sorted(list(set(semantic_ids)))
    assert unique_ids == list(range(len(unique_ids)))
    log_likelihood_per_semantic_id = []

    for uid in unique_ids:
        # Find positions in `semantic_ids` which belong to the active `uid`.
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        # Gather log likelihoods at these indices.
        id_log_likelihoods = [log_likelihoods[i] for i in id_indices]
        if agg == 'sum_normalized':
            # log_lik_norm = id_log_likelihoods - np.prod(log_likelihoods)
            log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
            logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        else:
            raise ValueError
        log_likelihood_per_semantic_id.append(logsumexp_value)

    return log_likelihood_per_semantic_id

def predictive_entropy(log_probs):
    """Compute MC estimate of entropy.

    `E[-log p(x)] ~= -1/N sum_i log p(x_i)`, i.e. the average token likelihood.
    """

    entropy = -np.sum(log_probs) / len(log_probs)

    return entropy

def predictive_entropy_rao(log_probs):
    entropy = -np.sum(np.exp(log_probs) * log_probs)
    return entropy

def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from how often different clusters get assigned.

    We estimate the categorical distribution over cluster assignments from the
    semantic ids. The uncertainty is then given by the entropy of that
    distribution. This estimate does not use token likelihoods, it relies soley
    on the cluster assignments. If probability mass is spread of between many
    clusters, entropy is larger. If probability mass is concentrated on a few
    clusters, entropy is small.

    Input:
        semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
    Output:
        cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
    """

    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts/n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy

