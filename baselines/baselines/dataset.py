from .utils import read_text, pad_or_trim_tensor

from typing import Dict, List, Union, Any
from pathlib import Path
import json
import pickle

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer



class PairedFinetuning(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer | None = None,
        max_len: int | None = 4096,
        add_bos_token: bool = True,
        mode: str = "finetune",
        use_target: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.add_bos_token = add_bos_token
        self.IGNORE_INDEX = -100
        self.pad_token_id = 2
        self.mode = mode
        self.use_target = use_target

        # 加载数据
        if Path(file_path).suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
        elif Path(file_path).suffix == '.pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        
        if not isinstance(data, dict):
            raise ValueError("文件格式不符合要求，应为字典结构")
        
        self.samples = self._process_data_for_finetuning(data)
        self.encoded = [self._tokenize_text(sample) for sample in self.samples]

    def _process_data_for_finetuning(self, data: Dict) -> List:
        samples = []
        if self.mode == "retrain":
            for sample_data in data.values():
                samples.append(sample_data['retain']['response'])
            if not samples:
                raise ValueError("未找到有效样本，请检查数据格式")
            return samples
        else:
            for sample_data in data.values():
                samples.append(sample_data['retain']['response'])
                if self.use_target:
                    samples.append(sample_data['target']['response'])
                if len(sample_data['forget']):
                    # 添加所有model_output的response
                    for case in sample_data['forget'].values():
                        if isinstance(case, dict) and 'response' in case:
                            samples.append(case['response'])
            if not samples:
                raise ValueError("未找到有效样本，请检查数据格式")
            return samples

    def _tokenize_text(self, text: str) -> torch.Tensor:
        encoding: torch.Tensor = self.tokenizer(
            text,
            add_special_tokens=self.add_bos_token,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
        )
        return encoding

    def __getitem__(self, index):
        labels = self.encoded[index]['input_ids'][0].clone()
        labels[labels == self.pad_token_id] = self.IGNORE_INDEX
        return {
            "input_ids": self.encoded[index]['input_ids'][0],
            "labels": labels,
            "attention_mask": self.encoded[index]['attention_mask'][0]
        }

    def __len__(self) -> int:
        return len(self.encoded)


class PairedUnlearning(Dataset):
    def __init__(
        self,
        file_path: str,
        retain: bool = False,
        tokenizer: AutoTokenizer | None = None,
        max_len: int | None = 4096,
        add_bos_token: bool = True,
        neg_sample_num: int = 0,
        use_target: bool = False,
    ):
        self.retain_exists = retain
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.add_bos_token = add_bos_token
        self.neg_sample_num = neg_sample_num
        self.IGNORE_INDEX = -100
        self.pad_token_id = 2
        self.use_target = use_target

        # 加载数据
        print("Loading from path: ", file_path)
        if Path(file_path).suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
        elif Path(file_path).suffix == '.pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        
        if not isinstance(data, dict):
            raise ValueError("文件格式不符合要求，应为字典结构")
        
        if self.retain_exists:
            self.pairs = self._process_dict_data(data)
            self.prepare_pairs()
        else:
            self.samples = self._process_dict_data(data)
            self.encoded = [self._tokenize_text(sample) for sample in self.samples]

    def _process_dict_data(self, data: Dict) -> List[Dict[str, Any]]:
        samples = []
        pairs = []
        for sample_data in data.values():
            if len(sample_data['forget']):
                generation = []
            if self.use_target:
                first_output = sample_data['target']['response']
            else:
                first_output = sample_data['retain']['response']

            for output in sample_data['forget'].values():
                    generation.append(output['response'])
                    samples.append(output['response'])

            pair = (first_output, generation)
            pairs.append(pair)

        if not generation:
            raise ValueError("未找到有效样本，请检查数据格式")
        if self.retain_exists:
            return pairs
        else:
            return samples

    def _tokenize_text(self, text: str) -> torch.Tensor:
        encoding: torch.Tensor = self.tokenizer(
            text,
            add_special_tokens=self.add_bos_token,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
        )
        return encoding

    def _tokenize_pair(self, pair: tuple[str, List[str]]) -> Dict[str, Any]:
        first_output, generation_list = pair
        tokenized_first = self._tokenize_text(first_output)
        tokenized_generations = [self._tokenize_text(gen) for gen in generation_list]
        return {
            'first_output': tokenized_first,
            'generations': tokenized_generations
        }
    
    def prepare_pairs(self):
        if hasattr(self, 'pairs'):
            self.tokenized_pairs = [self._tokenize_pair(pair) for pair in self.pairs]
        else:
            raise AttributeError("请先调用_process_dict_data方法生成pairs")
    
    def __getitem__(self, index):
        if self.retain_exists:
            tokenized_pair = self.tokenized_pairs[index]
            first_output = tokenized_pair['first_output']
            labels = first_output['input_ids'][0].clone()
            labels[labels == self.pad_token_id] = self.IGNORE_INDEX
            
            generations = []
            for gen in tokenized_pair['generations'][:self.neg_sample_num]:
                gen_labels = gen['input_ids'].clone()
                gen_labels[gen_labels == self.pad_token_id] = self.IGNORE_INDEX
                generations.append({
                    "input_ids": gen['input_ids'][0],
                    "labels": gen_labels,
                    "attention_mask": gen['attention_mask'][0]
                })
            return (
                {
                    "input_ids": first_output['input_ids'][0],
                    "labels": labels,
                    "attention_mask": first_output['attention_mask'][0],
                    "idx": index
                },
                generations
            )
        else: 
            input_ids = self.encoded[index]['input_ids'][0]
            labels = input_ids.clone()
            labels[labels == self.pad_token_id] = self.IGNORE_INDEX
            attention_mask = self.encoded[index]['attention_mask'][0]
            return (
                None,
                {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask
                }
            )

    def get_retain_raw_token_count(self) -> int:
        """统计 retain set 的原始 token 数"""
        if not self.retain_exists:
            raise ValueError("当前实例未加载 retain 数据。请设置 retain=True")
        
        total = 0
        for pair in self.pairs:
            retain_text = pair[0]
            tokens = self.tokenizer.tokenize(retain_text, add_special_tokens=self.add_bos_token)
            total += len(tokens)
        return total

    def get_forget_raw_token_count(self) -> int:
        """统计 forget set 的原始 token 数"""
        if not self.retain_exists:
            raise ValueError("当前实例未加载 retain 数据。请设置 retain=True")
        
        total = 0
        for pair in self.pairs:
            generations = pair[1]
            for text in generations:
                tokens = self.tokenizer.tokenize(text, add_special_tokens=self.add_bos_token)
                total += len(tokens)
        return total

    def __len__(self) -> int:
        if self.retain_exists:
            return len(self.pairs)
        else:
            return len(self.encoded)

    def get_collate_fn(self):
        def collate_fn(batch: List[tuple[Dict, List[Dict]]]):
            if self.retain_exists:
                retain_batch = [pair[0] for pair in batch]
                batch_retain_input_ids = torch.stack([item['input_ids'] for item in retain_batch])
                batch_retain_labels = torch.stack([item['labels'] for item in retain_batch])
                batch_retain_attention_mask = torch.stack([item['attention_mask'] for item in retain_batch])
                batch_Retain_idx = [item['idx'] for item in retain_batch]

                dict_retain = {
                    "input_ids": batch_retain_input_ids,
                    "labels": batch_retain_labels,
                    "attention_mask": batch_retain_attention_mask,
                    "idx": batch_Retain_idx
                }
                list_forget = []
                for pair in batch:
                    generations = pair[1]
                    selected = generations
                    list_forget.extend(selected)         
                batch_forget_input_ids = torch.stack([item['input_ids'] for item in list_forget])
                batch_forget_labels = torch.stack([item['labels'] for item in list_forget])
                batch_forget_attention_mask = torch.stack([item['attention_mask'] for item in list_forget])
                
                dict_forget = {
                    "input_ids": batch_forget_input_ids,
                    "labels": batch_forget_labels,
                    "attention_mask": batch_forget_attention_mask
                }
                return dict_retain, dict_forget
            else:
                forget_batch = [pair[1] for pair in batch]
                batch_forget_input_ids = torch.stack([item['input_ids'] for item in forget_batch])
                batch_forget_labels = torch.stack([item['labels'] for item in forget_batch])
                batch_forget_attention_mask = torch.stack([item['attention_mask'] for item in forget_batch])
                
                dict_forget = {
                    "input_ids": batch_forget_input_ids,
                    "labels": batch_forget_labels,
                    "attention_mask": batch_forget_attention_mask
                }
                return None, dict_forget
    
        return collate_fn

# unlearning and finetuning
class PairedDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        retain: bool = False,
        tokenizer: AutoTokenizer | None = None,
        max_len: int | None = 4096,
        add_bos_token: bool = True,
        neg_sample_num: int = 0,  # 此处的k决定了“负样本”的个数！
        use_target: bool = False, # 此处控制 模板文本 是否放入retain中; 当前暂时不控制此参数
        mode: str = None,
    ):
        """
        专用于处理pair数据的Dataset类
        数据结构要求:
        {
            "sample_id1": {
                "source_text": "原始文本",
                "target_text": "目标文本",
                "model_output": [
                    {"response": "模型生成1"},
                    {"response": "模型生成2"}
                ]
            },
            ...
        }
        """
        self.retain_exists = retain
        # assert tokenizer is not None, "Tokenizer must be specified."
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.add_bos_token = add_bos_token
        self.neg_sample_num = neg_sample_num
        self.mode = mode
        self.IGNORE_INDEX = -100  # 通常用于忽略的索引值
        self.pad_token_id = 2


        if Path(file_path).suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
        elif Path(file_path).suffix == '.pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        
        if not isinstance(data, dict):
            raise ValueError("文件格式不符合要求，应为字典结构")
        
        if mode == 'unlearning':
            if self.retain_exists:
                self.pairs = self._process_dict_data(data)
                self.prepare_pairs()
            else:
                self.samples = self._process_dict_data(data)
                self.encoded = [self._tokenize_text(sample) for sample in self.samples]
                
        elif mode == 'finetuning':
            self.samples = self._process_raw_data_for_finetuning(data)
            self.encoded = [self._tokenize_text(sample) for sample in self.samples]

    def _process_dict_data(self, data: Dict) -> List[Dict[str, Any]]:
        """处理字典结构数据，返回规范化的样本列表"""
        samples = []
        pairs = []
        for sample_data in data.values():  # 不需要sample_id，直接遍历值
            # 确保所有sample都有这两个key
            if not all(k in sample_data for k in ['source', 'target']):
                print(sample_data)
                break
            
            if len(sample_data['model_output']):
            # 构建generation列表
                generation = []
            
            cnt = 0
            # 添加所有model_output的response
            if 'model_output' in sample_data:
                for output in sample_data['model_output']:
                    if isinstance(output, dict) and 'response' in output:
                        # 暂时先确保每一条数据是整齐的
                        if cnt < 1:
                            first_output = output['response']
                        else:
                            generation.append(output['response'])
                        cnt += 1

            pair = (first_output, generation)
            pairs.append(pair)

            # 为了保障npo和cnpo的遗忘对象相同
            for gen in generation:
                samples.append(gen)
            # # 暂时不考虑target sample
            # generation.append(sample_data['target'])
            
        if not samples:
            raise ValueError("未找到有效样本，请检查数据格式")
        
        if self.retain_exists:
            return pairs
        else:
            return samples

    def _process_raw_data_for_finetuning(self, data:Dict)-> List:
        samples = []
        for sample_data in data.values():
            if len(sample_data['retain']):
                if len(sample_data['retain']) != 4:
                    continue
                # 添加所有model_output的response
                for case in sample_data['retain'].values():
                    
                    if isinstance(case, dict) and 'response' in case:
                        # 暂时先确保每一条数据是整齐的
                        samples.append(case['response'])
                        break
            
            # 不添加target进行finetuning
            # samples.append(sample_data['target'])
        if not samples:
            raise ValueError("未找到有效样本，请检查数据格式")
        return samples

    def _tokenize_text(self, text: str) -> torch.Tensor:
        """将文本tokenize并填充/修剪到max_len"""
        encoding: torch.Tensor = self.tokenizer(
            text,
            add_special_tokens=self.add_bos_token,
            return_tensors='pt',
            truncation=True,
            max_length=self.max_len,
            padding="max_length",  # 让tokenizer统一处理填充
        )
        
        # 因为我们的数据并没有那么长，因此2048足够了，不需要trim操作，因此把填充操作融合到上边
        # encoding = pad_or_trim_tensor(
        #             encoding,
        #             target_length=self.max_len,
        #             padding_value=self.tokenizer.pad_token_id
        # )
        return encoding

    def _tokenize_pair(self, pair: tuple[str, List[str]]) -> Dict[str, Any]:
        """
        Tokenize a pair of (first_output, generation_list)
        """
        first_output, generation_list = pair
        # Tokenize the first output
        tokenized_first = self._tokenize_text(first_output)
        
        # Tokenize each generation in the list
        tokenized_generations = [self._tokenize_text(gen) for gen in generation_list]
        
        return {
            'first_output': tokenized_first,
            'generations': tokenized_generations
        }
    
    def prepare_pairs(self):
        """
        Tokenize all pairs and store them
        """
        if hasattr(self, 'pairs'):
            self.tokenized_pairs = [self._tokenize_pair(pair) for pair in self.pairs]
        else:
            raise AttributeError("请先调用_process_dict_data方法生成pairs")
    
    def __getitem__(self, index):
        if self.mode == 'finetuning':
            labels = self.encoded[index]['input_ids'][0].clone()
            labels[labels == self.pad_token_id] = self.IGNORE_INDEX
            return {"input_ids": self.encoded[index]['input_ids'][0],
                    "labels": labels,
                    "attention_mask": self.encoded[index]['attention_mask'][0]
                }
        
        else:
            if self.retain_exists:
                tokenized_pair = self.tokenized_pairs[index]
            else: 
                input_ids = self.encoded[index]['input_ids'][0]
                labels = input_ids.clone()
                labels[labels == self.pad_token_id] = self.IGNORE_INDEX
                attention_mask = self.encoded[index]['attention_mask'][0]
                return (
                    None,
                    {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention_mask
                    }
                )
                
            first_output = tokenized_pair['first_output']
            labels = first_output['input_ids'][0].clone()
            labels[labels == self.pad_token_id] = self.IGNORE_INDEX
            
            generations = []
            for gen in tokenized_pair['generations'][:self.neg_sample_num]:
                gen_labels = gen['input_ids'].clone()
                gen_labels[gen_labels == self.pad_token_id] = self.IGNORE_INDEX
                generations.append({
                    "input_ids": gen['input_ids'][0],
                    "labels": gen_labels,
                    "attention_mask": gen['attention_mask'][0]
                })
        
            return (
                {  # 第一个输出的数据
                    "input_ids": first_output['input_ids'][0],
                    "labels": labels,
                    "attention_mask": first_output['attention_mask'][0]
                },
                generations  # 生成列表的数据
            )

    def __len__(self) -> int:
        if self.retain_exists:
            return len(self.pairs)
        else:
            return len(self.encoded)

    def get_collate_fn(self):
        """获取处理批量的函数，适配pair数据结构"""
        def collate_fn(batch: List[tuple[Dict, List[Dict]]]):
            # 处理第一个输出 (forget)
            if self.retain_exists:
                retain_batch = [pair[0] for pair in batch]  # 获取所有first_output
                batch_retain_input_ids = torch.stack([item['input_ids'] for item in retain_batch])
                batch_retain_labels = torch.stack([item['labels'] for item in retain_batch])
                batch_retain_attention_mask = torch.stack([item['attention_mask'] for item in retain_batch])

                dict_retain = {
                    "input_ids": batch_retain_input_ids,
                    "labels": batch_retain_labels,
                    "attention_mask": batch_retain_attention_mask
                }

                list_forget = []
                for pair in batch:
                    generations = pair[1]  # 获取该样本的generations列表
                    selected = generations
                    list_forget.extend(selected)
                
                # 堆叠所有retain样本
                batch_forget_input_ids = torch.stack([item['input_ids'] for item in list_forget])
                batch_forget_labels = torch.stack([item['labels'] for item in list_forget])
                batch_forget_attention_mask = torch.stack([item['attention_mask'] for item in list_forget])
                
                dict_forget = {
                    "input_ids": batch_forget_input_ids,
                    "labels": batch_forget_labels,
                    "attention_mask": batch_forget_attention_mask
                }
                return dict_retain, dict_forget
            else:
                forget_batch = [pair[1] for pair in batch]  # 获取所有first_output
                batch_forget_input_ids = torch.stack([item['input_ids'] for item in forget_batch])
                batch_forget_labels = torch.stack([item['labels'] for item in forget_batch])
                batch_forget_attention_mask = torch.stack([item['attention_mask'] for item in forget_batch])
                
                dict_forget = {
                    "input_ids": batch_forget_input_ids,
                    "labels": batch_forget_labels,
                    "attention_mask": batch_forget_attention_mask
                }

                # 处理生成列表 (retain)
                return None, dict_forget
    
        return collate_fn



class DefaultDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        tokenizer: AutoTokenizer | None = None,
        max_len: int | None = 4096,
        add_bos_token: bool = True
    ):
        if Path(file_path).suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
            if isinstance(data[0], str):
                self.strings = data
            elif isinstance(data[0], dict) and 'text' in data[0] \
                    and isinstance(data[0]['text'], str):
                self.strings = [d['text'] for d in data]
                if 'input_ids' in data[0]:
                    self.input_ids = [torch.tensor(d['input_ids']) for d in data]
                    return; # Done, since we have `input_ids` ready.
            else:
                raise ValueError("Format of this `.json` file is not recognized.")

            assert tokenizer is not None, "Tokenizer must be specified."

            self.input_ids = []
            for s in self.strings:
                encoding: torch.Tensor = tokenizer(
                    s,
                    add_special_tokens=add_bos_token,
                    return_tensors='pt'
                ).input_ids[0]
                encoding = pad_or_trim_tensor(
                    encoding,
                    target_length=max_len,
                    padding_value=tokenizer.pad_token_id
                )
                self.input_ids.append(encoding)

            return; # end if Path(file_path).suffix == '.json'        

        assert Path(file_path).suffix == '.txt'

        tokens = tokenizer(read_text(file_path), add_special_tokens=False, return_tensors='pt').input_ids[0]
        assert len(tokens.shape) == 1, "Debug error: Tokens not 1-dimensional"

        if add_bos_token:
            self.input_ids = [
                F.pad(
                    tokens[i : i + max_len - 1], (1, 0),
                    value=tokenizer.bos_token_id
                )
                for i in range(0, len(tokens), max_len - 1)
            ]
        else:
            self.input_ids = [
                tokens[i : i + max_len]
                for i in range(0, len(tokens), max_len)
            ]

        # Rotate the tokens if the last `input_ids` isn't filled to max_len
        if len(self.input_ids[-1]) < max_len:
            self.input_ids[-1] = torch.concat(
                [self.input_ids[-1], self.input_ids[0]], dim=-1
            )[:max_len]

        # Original strings
        self.strings = tokenizer.batch_decode(self.input_ids, skip_special_tokens=True)
        pass    # def __init__()


    def __getitem__(self, index):
        return self.input_ids[index]


    def __len__(self):
        return len(self.input_ids)


    def get_collate_fn(self):

        def collate_fn(batch: List[torch.Tensor]):
            batch = torch.stack(batch)
            return {
                "input_ids": batch,
                "labels": batch.clone()
            }

        return collate_fn

class ForgetRetainDataset(DefaultDataset):

    def __init__(
        self,
        forget_file_path: str,
        tokenizer: AutoTokenizer,
        retain_file_path: str | None = None,
        max_len: int = 4096,
        add_bos_token: bool = True
    ):
        self.forget_dataset = DefaultDataset(
            forget_file_path, tokenizer,
            max_len=max_len, add_bos_token=add_bos_token
        )

        self.retain_exists = retain_file_path is not None
        if self.retain_exists:
            self.retain_dataset = DefaultDataset(
                retain_file_path, tokenizer,
                max_len=max_len, add_bos_token=add_bos_token
            )

        self.tokenizer = tokenizer


    def __getitem__(self, index):
        if self.retain_exists:
            return (
                self.retain_dataset[index % len(self.retain_dataset)],
                self.forget_dataset[index],
            )
        else:
            return None, self.forget_dataset[index]


    def __len__(self):
        return len(self.forget_dataset)


    def get_collate_fn(self):

        def collate_fn(batch: List[tuple[torch.Tensor, torch.Tensor]]):
            if self.retain_exists:
                batch_retain = torch.stack([pair[0] for pair in batch])
                dict_retain = {
                    "input_ids": batch_retain,
                    "labels": batch_retain.clone(),
                    "attention_mask": torch.ones_like(batch_retain, dtype=torch.bool)
                }
            else:
                dict_retain = None

            batch_forget = torch.stack([pair[1] for pair in batch])
            dict_forget = {
                "input_ids": batch_forget,
                "labels": batch_forget.clone(),
                "attention_mask": torch.ones_like(batch_forget)
            }
            return dict_retain, dict_forget
        return collate_fn

class ContrastiveDataset(DefaultDataset):
    def __init__(
            self,
            forget_file_path: str,
            tokenizer: AutoTokenizer,
            retain_file_path: str | None = None,
            max_len: int = 4096,
            add_bos_token: bool = True,
            neg_sample_num: int = 0,  # 此处的k决定了“负样本”的个数！
            version: int = 3,
    ):
        self.neg_sample_num = neg_sample_num
        self.version = version

        self.forget_dataset = DefaultDataset(
            forget_file_path, tokenizer,
            max_len=max_len, add_bos_token=add_bos_token
        )

        self.retain_exists = retain_file_path is not None
        if self.retain_exists:
            self.retain_dataset = DefaultDataset(
                retain_file_path, tokenizer,
                max_len=max_len, add_bos_token=add_bos_token
            )

        self.tokenizer = tokenizer

    def __getitem__(self, index):
        if self.version == 3:
            if self.retain_exists and self.neg_sample_num != 0:
                k = self.neg_sample_num
                max_offset = len(self.forget_dataset) - 1
                offsets = [0]
                while len(offsets) < k:
                    rand_offset = torch.randint(1, max_offset, (1,)).item()  # 生成1~max_offset-1的随机数
                    if rand_offset not in offsets:
                        offsets.append(rand_offset)
                forget_samples = []
                for offset in offsets:
                    forget_idx = (index + offset) % len(self.forget_dataset)
                    forget_item = self.forget_dataset[forget_idx]
                    forget_samples.append(forget_item)
                
                return (self.retain_dataset[index % len(self.retain_dataset)], forget_samples)          
        else:
            if self.retain_exists and self.neg_sample_num != 0:
                return (
                    # 此处的index模retain_dataset的长度以防止越界
                    torch.stack(self.retain_dataset[index % len(self.retain_dataset) : (index + self.neg_sample_num) % len(self.retain_dataset)]),
                    self.forget_dataset[index],
                )
            else:
                return None, self.forget_dataset[index]

    def __len__(self):
        return len(self.forget_dataset)

    def get_collate_fn(self):

        def collate_fn(batch: List[tuple[torch.Tensor, torch.Tensor]]):
            if self.version == 3:
                # 此处可能需要修改：因为labels, attention_mask可能不一样，需要进入trainer检查
                batch_retain = torch.stack([pair[0] for pair in batch])
                dict_retain = {
                    "input_ids": batch_retain,
                    "labels": batch_retain.clone(),
                    "attention_mask": torch.ones_like(batch_retain, dtype=torch.bool)
                }
                batch_forget = torch.stack([tensor for pair in batch for tensor in pair[1]])
                # 此处可能需要修改：因为labels, attention_mask可能不一样，需要进入trainer检查
                dict_forget = {
                    "input_ids": batch_forget,
                    "labels": batch_forget.clone(),
                    "attention_mask": torch.ones_like(batch_forget)
                }
            else:
                batch_forget = torch.stack([pair[1] for pair in batch])
                dict_forget = {
                    "input_ids": batch_forget,
                    "labels": batch_forget.clone(),
                    "attention_mask": torch.ones_like(batch_forget)
                }

                if self.retain_exists:
                    batch_retain = torch.stack([tensor for pair in batch for tensor in pair[0]])
                    dict_retain = {
                        "input_ids": batch_retain,
                        "labels": batch_retain.clone(),
                        "attention_mask": torch.ones_like(batch_retain, dtype=torch.bool)
                    }
                else:
                    dict_retain = None

            return dict_retain, dict_forget 

        return collate_fn