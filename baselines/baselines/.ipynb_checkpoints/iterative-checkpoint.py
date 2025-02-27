from scipy.ndimage import label

from .utils import load_model_and_tokenizer, load_model
from .dataset import ForgetRetainDataset, ContrastiveDataset

import torch
import torch.nn.functional as F
from torch.cuda import device_count
import transformers
from transformers import Trainer, AutoModelForCausalLM
from peft import  get_peft_model, LoraConfig

def unlearn(
    model_dir: str,
    data_file: str,
    out_dir: str,
    retain_data_file: str | None = None,
    loss_type: str = 'ga',
    per_device_batch_size: int = 2,
    epochs: int = 5,
    learning_rate=1e-5,
    max_len: int = 4096,
    tokenizer_dir: str | None = None,
    resume_from_checkpoint: bool = False,

    neg_sample_num: int = 2,
    alpha : float = 1,
    coeff_type : str | None = None,
    use_lora : bool | None = False,
):
    if 'gd' in loss_type:
        assert retain_data_file is not None, "Retain data must be specified for grad_diff."

    model, tokenizer = load_model_and_tokenizer(
        model_dir,
        tokenizer_dir=tokenizer_dir
    )
    model.enable_input_require_grads()
    print("Using algorithm: ", loss_type)

    ref_model = (
        load_model(model_dir)
        if 'npo' in loss_type or 'kl' in loss_type or 'cont_npo' in loss_type
        else None
    )
    
    if use_lora:
        peft_config = LoraConfig(
            r=8, 
            lora_alpha=32, 
            target_modules=["query_key_value"], 
            lora_dropout=0.05,
            bias="none", 
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        for name, param in model.base_model.named_parameters():
            if name is not None and ("lora_A" in name or "lora_B" in name):  # 确保 name 不为 None
                param.requires_grad = True
            else:
                param.requires_grad = False
    

    # dataset = ForgetRetainDataset(
    #     data_file,
    #     tokenizer=tokenizer,
    #     retain_file_path=retain_data_file,
    #     max_len=max_len
    # )

    dataset = ContrastiveDataset(
        data_file,
        tokenizer=tokenizer,
        retain_file_path=retain_data_file,
        max_len=max_len,
        neg_sample_num=neg_sample_num, #k是负样本的个数
    )

    if device_count() == 0:
        raise ValueError("Device not detected!")

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        save_strategy='epoch',  # Save every epoch
        num_train_epochs=epochs,
        optim='adamw_torch',
        # optim="adamw_bnb_8bit", #尝试用牺牲精度方式来减小显存
        gradient_checkpointing=True, # 激活梯度检查点 # try
        lr_scheduler_type='constant',
        bf16=True,
        report_to='none',  # Disable wandb
        # ddp_find_unused_parameters=False,  # 关闭 DDP 查找未使用参数
    )


    trainer = IterativeUnlearner(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=dataset.get_collate_fn(),
        loss_type=loss_type,
        alpha=alpha, #额外添加
        neg_sample_num=neg_sample_num, #额外添加
        coeff_type=coeff_type, #额外添加
    )
    
    model.config.use_cache = False  # silence the warnings.
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(out_dir)



class IterativeUnlearner(Trainer):
    """Source: https://github.com/locuslab/tofu/blob/main/dataloader.py
    """

    def __init__(self, *args,
                 loss_type: str = 'cont_npo',
                 ref_model: AutoModelForCausalLM | None = None,
                 beta: float = 0.1,
                 neg_sample_num=2, #额外添加
                 alpha: float = 1, #额外添加
                 coeff_type: str = 'cosine', #额外添加
                 **kwargs):
        self.loss_type = loss_type
        self.ref_model = ref_model
        self.beta = beta # Only relevant when `'po' in self.loss_type`
        self.alpha = alpha #额外添加
        self.neg_sample_num = neg_sample_num #额外添加
        self.coeff_type = coeff_type #额外添加

        if ref_model is not None:
            assert 'po' in self.loss_type or 'kl' in self.loss_type
            ref_model = ref_model.eval()

        super().__init__(*args, **kwargs)


    def compute_loss(self, model, x, return_outputs=False):
        """Source: https://github.com/licong-lin/negative-preference-optimization/blob/main/synthetic/mymodel.py
        """
        model.train()
        ### 1. Run model ###
        x_f, x_r = x
        
        outputs_f = model(
            x_f['input_ids'],
            labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
            attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool),
            output_hidden_states = True,
            # output_attentions=False,
        )
        loss_f = outputs_f.loss

        if 'gdr' in self.loss_type or 'klr' in self.loss_type:
            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
            )
            loss_r = outputs_r.loss

        if 'klf' in self.loss_type or 'npo' == self.loss_type:
            with torch.no_grad():
                outputs_f_ref = self.ref_model(
                    x_f['input_ids'],
                    labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                    attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool)
                )

        if 'klr' in self.loss_type:
            with torch.no_grad():
                outputs_r_ref = self.ref_model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool)
                )

        if 'cont_npo' in self.loss_type:
            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool),
                output_hidden_states=True
                # output_attentions=False,
            )
            loss_r = outputs_r.loss

            with torch.no_grad():
                outputs_r_ref = self.ref_model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(
                        x_r['input_ids'], dtype=torch.bool),
                )

                outputs_f_ref = self.ref_model(
                    x_f['input_ids'],
                    labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                    attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool),
                )

        ### 2. Compute Loss ###
        loss = 0
        # print("\nComputing loss: ","Algo: ", self.loss_type)

        if 'ga' in self.loss_type:
            loss += -loss_f

        elif 'npo' == self.loss_type:
            neg_log_ratio = outputs_f_ref.logits - outputs_f.logits
            loss += -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta

        # todo: 此处并未解决一个问题：即如何使k和x_f的shape[0]不同
        # todo: 因为对比学习需要选取正负样本对，那么是否在loss的提取有所不同呢？ 况且由于存在样本的选取问题，data的loading过程是否需要更改？
        
        elif 'cont_npo' in self.loss_type:
            from math import log, exp
            import ipdb
            total_coeff = []
            k = self.neg_sample_num

            if self.coeff_type == 'cosine':
                # 计算余弦相似度
                with torch.no_grad():
                    # 保持计算在GPU上，不要将数据转移到CPU
                    embeddings_f = outputs_f.hidden_states[-1][:, -1, :]
                    embeddings_r = outputs_r.hidden_states[-1][:, -1, :]

                # for idx in range(x_f['input_ids'].shape[0]):
                #     temp_sum = 0
                #     for j in range(x_r['input_ids'].shape[0]):
                #         # 计算余弦相似度的dot product和norm
                #         cos_similarity = torch.nn.functional.cosine_similarity(embeddings_f[j].unsqueeze(0), embeddings_r[idx].unsqueeze(0))
                #         temp_sum += exp((1 - cos_similarity) / self.alpha)
                #     total_coeff.append(temp_sum)

                for idx in range(x_f['input_ids'].shape[0]):
                    temp_sum = 0
                    for j in range(k):
                        # 计算余弦相似度的dot product和norm
                        cos_similarity = torch.nn.functional.cosine_similarity(embeddings_f[idx].unsqueeze(0), embeddings_r[idx + j].unsqueeze(0))
                        temp_sum += cos_similarity / self.alpha
                    total_coeff.append(temp_sum)
            elif self.coeff_type == 'semantic_entropy':
                from .semantic_entropy import EntailmentPythia
                pythia = EntailmentPythia(local_model_path='/mnt/wenjt5/muse/model/pythia/pythia-410m-news')
                for idx in range(x_f['input_ids'].shape[0]):
                    temp_sum = 0
                    for j in range(x_r['input_ids'].shape[0]):
                        input_ids = torch.cat((x_f['input_ids'][idx].unsqueeze(0),
                                               x_r['input_ids'][j].unsqueeze(0)),dim=0)
                        semantic_entropy = pythia.compute_semantic_entropy(input_ids=input_ids)
                        temp_sum += exp(semantic_entropy / self.alpha)

            # 更新loss计算，避免重复计算
            for idx in range(x_f['input_ids'].shape[0]):
                for j in range(k):
                    log_ratio_1 = outputs_r.logits[idx + j] - outputs_r_ref.logits[idx + j] - log(k)
                    log_ratio_2 = outputs_f_ref.logits[idx] - outputs_f.logits[idx] + log(k)

                    if self.coeff_type == 'cosine':
                        # 计算余弦相似度
                        coeff = torch.nn.functional.cosine_similarity(embeddings_f[idx].unsqueeze(0), embeddings_r[idx + j].unsqueeze(0))
                        temp1 = (exp((1 - coeff) / self.alpha) / total_coeff[idx]) * F.logsigmoid(log_ratio_1)
                        # temp1 = ((coeff / self.alpha) / total_coeff[idx]) * F.logsigmoid(log_ratio_1)
                    elif self.coeff_type == 'semantic_entropy':
                        # coeff = compute_semantic_entropy(torch.cat((x_f['input_ids'][idx],x_r['input_ids'][j]),dim=0))
                        temp1 = (exp(coeff / self.alpha) / total_coeff[idx]) * F.logsigmoid(log_ratio_1)
                    elif self.coeff_type == 'distance':
                        return
                    temp2 = F.logsigmoid(log_ratio_2) / k
                    loss += temp1 + temp2

            # 归一化损失
            loss = -loss.mean() / (x_f['input_ids'].shape[0] * x_r['input_ids'].shape[0])

        else:
            raise NotImplementedError("Cannot infer the given loss type.")

        if 'gdr' in self.loss_type:
            loss += loss_r

        if 'klf' in self.loss_type:
            raise NotImplementedError("KL forget not implemented yet!")

        if 'klr' in self.loss_type:
            kl_r = F.kl_div(
                outputs_r.logits,
                outputs_r_ref.logits,
                reduction = 'batchmean',
                log_target = True
            )
            loss += kl_r

        return (loss, outputs_f) if return_outputs else loss


    def prediction_step(self, model, x, prediction_loss_only: bool, ignore_keys=None):
        input_ids, labels, attention_mask = x
        # forward pass
        with torch.no_grad():
            outputs = model(input_ids, labels=labels, attention_mask=attention_mask)
            logits = outputs.logits
            loss = outputs.loss
        return (loss, logits, labels)
