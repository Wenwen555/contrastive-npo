from .utils import read_text, pad_or_trim_tensor

from typing import Dict, List, Union, Any
from pathlib import Path
import json
import pickle

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import AutoTokenizer


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
        version: int = 3,
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
        self.version = version,

        if Path(file_path).suffix == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
        elif Path(file_path).suffix == '.pkl':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        
        if not isinstance(data, dict):
            raise ValueError("文件格式不符合要求，应为字典结构")
        
        if mode == 'unlearning':
            self.samples = self._process_dict_data(data)
            self.input_ids = []
            for sample in self.samples:
                ids = {}
                ids['source_ids'] = self._tokenize_text(sample['source_text'])
                ids['generation_ids'] = [self._tokenize_text(response) for response in sample['generation']]
                self.input_ids.append(ids)
        elif mode == 'finetuning':
            self.samples = self._process_raw_data_for_finetuning(data)
            self.input_ids = torch.stack([self._tokenize_text(sample) for sample in self.samples])

    def _process_dict_data(self, data: Dict) -> List[Dict[str, Any]]:
        """处理字典结构数据，返回规范化的样本列表"""
        samples = []
        
        for sample_data in data.values():  # 不需要sample_id，直接遍历值
            # 确保所有sample都有这两个key
            if not all(k in sample_data for k in ['source', 'target']):
                print(sample_data)
                break
            
            if len(sample_data['model_output']):
            # 构建generation列表
              generation = []
            
            # 添加所有model_output的response
            if len(sample_data['model_output']) <= 3:
                continue
            if 'model_output' in sample_data:
                for output in sample_data['model_output']:
                    if isinstance(output, dict) and 'response' in output:
                        # 暂时先确保每一条数据是整齐的
                        generation.append(output['response'])
            
            # 确保至少有一个generation（target_text）
            generation.append(sample_data['target'])
            
            samples.append({
                'source_text': sample_data['source'],
                'generation': generation
            })
        if not samples:
            raise ValueError("未找到有效样本，请检查数据格式")
            
        return samples

    def _process_raw_data_for_finetuning(self, data:Dict)-> List:
        samples = []
        for sample_data in data.values():  # 不需要sample_id，直接遍历值
            # 确保所有sample都有这两个key
            if not all(k in sample_data for k in ['source', 'target']):
                print(sample_data)
                break
            
            if len(sample_data['model_output']):
                if len(sample_data['model_output']) != 4:
                    continue
                # 添加所有model_output的response
                if 'model_output' in sample_data:
                    for output in sample_data['model_output']:
                        if isinstance(output, dict) and 'response' in output:
                            # 暂时先确保每一条数据是整齐的
                            samples.append(output['response'])
            
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
            max_length=self.max_len
        ).input_ids[0]
        
        encoding = pad_or_trim_tensor(
                    encoding,
                    target_length=self.max_len,
                    padding_value=self.tokenizer.pad_token_id
        )
        return encoding

    def __getitem__(self, index):
        if self.mode == 'finetuning':
            return {"input_ids": self.input_ids[index],
                    "labels": self.input_ids[index],
                    "attention_mask": torch.ones_like(self.input_ids[index])
                }  # 返回 dict
        else:
            ids = self.input_ids[index]
            return (
                ids["source_ids"],
                ids["generation_ids"]
            )
    # def __getitem__(self, index) -> Dict[str, Union[str, List[str]]]:
    #     """返回一个样本字典，包含:
    #     - source_text: str
    #     - generation: List[str] (包含模型输出和最后的target_text)
    #     """
    #     # 此处需要修改返回值为input_ids，否则get_collate_fn的输入为空
    #     return self.input_ids[index]

    def __len__(self) -> int:
        return len(self.samples)

    def get_collate_fn(self):
        """获取处理批量的函数，可根据需要自定义"""
        def collate_fn(batch: List[Dict[str, Union[str, List[str]]]]):
            source_texts = [pair[0] for pair in batch]
            batch_forget = torch.stack(source_texts)
            
            if self.retain_exists:
                list_retain = []
                for sample in batch:
                    cnt = 0
                    for item in sample[1]:
                        if cnt < self.neg_sample_num:
                            list_retain.append(item)
                            cnt += 1
                batch_retain = torch.stack(list_retain)
                dict_retain = {
                "input_ids": batch_retain,
                "labels": batch_retain.clone(),
                "attention_mask": torch.ones_like(batch_forget)
            }

            dict_forget = {
                "input_ids": batch_forget,
                "labels": batch_forget.clone(),
                "attention_mask": torch.ones_like(batch_forget)
            }

            
            if self.retain_exists:
                return dict_forget, dict_retain
            else:
                return dict_forget, None
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
        # todo: 需要弄清此处的rotate作用！
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
                self.forget_dataset[index],
                self.retain_dataset[index % len(self.retain_dataset)], #todo: 为何此处的index要模retain_dataset的长度呢？
            )
        else:
            return self.forget_dataset[index], None


    def __len__(self):
        return len(self.forget_dataset)


    def get_collate_fn(self):

        def collate_fn(batch: List[tuple[torch.Tensor, torch.Tensor]]):
            batch_forget = torch.stack([pair[0] for pair in batch])
            dict_forget = {
                "input_ids": batch_forget,
                "labels": batch_forget.clone(),
                "attention_mask": torch.ones_like(batch_forget)
            }

            if self.retain_exists:
                batch_retain = torch.stack([pair[1] for pair in batch])
                dict_retain = {
                    "input_ids": batch_retain,
                    "labels": batch_retain.clone(),
                    "attention_mask": torch.ones_like(batch_retain, dtype=torch.bool)
                }
            else:
                dict_retain = None

            return dict_forget, dict_retain

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
                # 此处的防越界需要重新处理
                start = index % len(self.forget_dataset)
                end = (index + self.neg_sample_num) % len(self.forget_dataset)
                if start > end:
                    forget_list = self.forget_dataset[start : len(self.forget_dataset)]
                    for i in range(end):
                        forget_list.append(self.forget_dataset[i]) 
                    forget_samples = torch.stack(forget_list)
                    return (
                        forget_samples,
                        self.retain_dataset[index],
                    )
                else:
                    return (
                        torch.stack(self.forget_dataset[index % len(self.forget_dataset) : (index + self.neg_sample_num) % len(self.forget_dataset)]),
                        self.retain_dataset[index],
                    )
        else:
            if self.retain_exists and self.neg_sample_num != 0:
                return (
                    self.forget_dataset[index],
                    # 此处的index模retain_dataset的长度以防止越界
                    torch.stack(self.retain_dataset[index % len(self.retain_dataset) : (index + self.neg_sample_num) % len(self.retain_dataset)]),
                )
            else:
                return self.forget_dataset[index], None

    def __len__(self):
        return len(self.forget_dataset)

    def get_collate_fn(self):

        def collate_fn(batch: List[tuple[torch.Tensor, torch.Tensor]]):
            if self.version == 3:
                batch_forget = torch.stack([tensor for pair in batch for tensor in pair[0]])
                dict_forget = {
                    "input_ids": batch_forget,
                    "labels": batch_forget.clone(),
                    "attention_mask": torch.ones_like(batch_forget)
                }

                if self.retain_exists:
                    batch_retain = torch.stack([pair[1] for pair in batch])
                    dict_retain = {
                        "input_ids": batch_retain,
                        "labels": batch_retain.clone(),
                        "attention_mask": torch.ones_like(batch_retain, dtype=torch.bool)
                    }
                else:
                    dict_retain = None
                    
            else:
                batch_forget = torch.stack([pair[0] for pair in batch])
                dict_forget = {
                    "input_ids": batch_forget,
                    "labels": batch_forget.clone(),
                    "attention_mask": torch.ones_like(batch_forget)
                }

                if self.retain_exists:
                    # batch_retain = torch.stack([pair[1] for pair in batch])
                    # batch_retain = torch.stack([torch.stack(pair[1]) for pair in batch])
                    batch_retain = torch.stack([tensor for pair in batch for tensor in pair[1]])
                    dict_retain = {
                        "input_ids": batch_retain,
                        "labels": batch_retain.clone(),
                        "attention_mask": torch.ones_like(batch_retain, dtype=torch.bool)
                    }
                else:
                    dict_retain = None

            return dict_forget, dict_retain

        return collate_fn