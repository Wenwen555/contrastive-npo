from scipy.ndimage import label

from .utils import load_model_and_tokenizer, load_model
from .dataset import ContrastiveDataset, PairedDataset
from math import log, exp

import torch
import torch.nn.functional as F
from torch.cuda import device_count
import transformers
from peft import  get_peft_model, LoraConfig
from transformers import Trainer, AutoModelForCausalLM
import warnings
import wandb

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
    use_lora : bool | None = True,
    version : int = 3,
):
    if 'pii' not in data_file:
        if 'gd' in loss_type:
            # this is not valid for pii data!
            assert retain_data_file is not None, "Retain data must be specified for grad_diff."

    if use_lora:
        print("using lora for finetuning!")
        base_model, tokenizer = load_model_and_tokenizer(
            tokenizer_dir,
            tokenizer_dir=tokenizer_dir
        )
        base_model.enable_input_require_grads()
        model = PeftModel.from_pretrained(base_model, model_dir)
        model.print_trainable_parameters()

        ref_model = (
            load_model(
                model_dir = model_dir,
                base_model_dir=tokenizer_dir,
                use_lora=use_lora
            )
            if 'npo' in loss_type or 'kl' in loss_type or 'cont_npo' in loss_type
            else None
        )
    else:
        # 不用peft
        print("using original model!!")
        model, tokenizer = load_model_and_tokenizer(
            model_dir,
            tokenizer_dir=tokenizer_dir
        )
        model.enable_input_require_grads()

        ref_model = (
            load_model(model_dir)
            if 'npo' in loss_type or 'kl' in loss_type or 'cont_npo' in loss_type
            else None
        )

    print("Using algorithm: ", loss_type)

    use_lora = True
    if use_lora:
        peft_config = LoraConfig(
            r=32, 
            lora_alpha=64, 
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], 
            lora_dropout=0.05,
            bias="none", 
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    

    # News and Books using!
    dataset = ContrastiveDataset(
        data_file,
        tokenizer=tokenizer,
        retain_file_path=retain_data_file,
        max_len=max_len,
        neg_sample_num=neg_sample_num,
        version=version,
    )
    
    # if loss_type == 'npo':
    #     retain = False
    # else:
    #     retain = True

    # dataset = PairedDataset(
    #     data_file,
    #     tokenizer=tokenizer,
    #     max_len=max_len,
    #     neg_sample_num=neg_sample_num, #k是负样本的个数
    #     mode='unlearning',
    #     retain=retain,
    # )

    if device_count() == 0:
        raise ValueError("Device not detected!")

    wandb.init(project="MUSE-7B-pii", name="-".join([loss_type,coeff_type,str(neg_sample_num),f'v{version}','lora']))

    wandb.watch(
        model,  # 要监视的模型
        log="all",  # 记录梯度 ("gradients") 和参数 ("parameters")
        log_freq=10,  # 每 10 步记录一次
    )

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        save_strategy='epoch',  # Save every epoch
        num_train_epochs=epochs,
        optim='adamw_torch',
        # gradient_checkpointing=True, # 激活梯度检查点 # try
        lr_scheduler_type='constant',
        bf16=True,
        report_to='wandb',  # Disable wandb
        # ddp_find_unused_parameters=False,  # 关闭 DDP 查找未使用参数
        # gradient_accumulation_steps = 0,
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
        version=version, #额外添加
    )

    warnings.filterwarnings("ignore", category=UserWarning)
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
                 version: int = 3, #额外添加:默认用第三种version loss function
                 **kwargs):
        self.loss_type = loss_type
        self.ref_model = ref_model
        self.beta = beta # Only relevant when `'po' in self.loss_type`
        self.alpha = alpha #额外添加
        self.neg_sample_num = neg_sample_num #额外添加
        self.coeff_type = coeff_type #额外添加
        self.version = version #额外添加

        if ref_model is not None:
            assert 'po' in self.loss_type or 'kl' in self.loss_type
            ref_model = ref_model.eval()
        
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, x, num_items_in_batch=None, return_outputs=False):
        """Source: https://github.com/licong-lin/negative-preference-optimization/blob/main/synthetic/mymodel.py
        """
        model.train()
        ### 1. Run model ###
        if self.version != 3:
            x_f, x_r = x
        else:
            x_r, x_f = x
        
        outputs_f = model(
            x_f['input_ids'],
            labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
            attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(x_f['input_ids'], dtype=torch.bool),
            output_hidden_states=True,
        )
        loss_f = outputs_f.loss

        if 'cont_npo' in self.loss_type:
            
            outputs_r = model(
                x_r['input_ids'],
                labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(x_r['input_ids'], dtype=torch.bool),
                output_hidden_states=True,
            )
            loss_r = outputs_r.loss

            with torch.no_grad():
                outputs_r_ref = self.ref_model(
                    x_r['input_ids'],
                    labels=x_r['labels'] if 'labels' in x_r else x_r['input_ids'].clone(),
                    attention_mask=x_r['attention_mask'] if 'attention_mask' in x_r else torch.ones_like(
                        x_r['input_ids'], dtype=torch.bool),
                    # output_hidden_states=True,
                )

                outputs_f_ref = self.ref_model(
                    x_f['input_ids'],
                    labels=x_f['labels'] if 'labels' in x_f else x_f['input_ids'].clone(),
                    attention_mask=x_f['attention_mask'] if 'attention_mask' in x_f else torch.ones_like(
                        x_f['input_ids'], dtype=torch.bool),
                    # output_hidden_states=True,
                )
        else:
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


        ### 2. Compute Loss ###
        loss = 0

        if 'ga' in self.loss_type:
            loss += -loss_f

        elif self.loss_type in ['npo', 'npo_klr', 'npo_gdr']:
            neg_log_ratio = outputs_f_ref.logits - outputs_f.logits
            loss += -F.logsigmoid(self.beta * neg_log_ratio).mean() * 2 / self.beta
            wandb.log({"loss": loss.item()}) 
            if torch.isnan(loss):
                print("NaN detected in loss") 
                print("total coeff is: ", total_coeff)
        
        elif 'cont_npo' in self.loss_type:
            total_coeff = []
            k = self.neg_sample_num
            version = self.version

            # what if treating it as a changing term when unlearning?
            # with torch.no_grad():
            #     embeddings_f = outputs_f_ref.hidden_states[-1][:, -1, :]
            #     embeddings_r = outputs_r_ref.hidden_states[-1][:, -1, :]

            embeddings_f = outputs_f.hidden_states[-1][:, -1, :]
            embeddings_r = outputs_r.hidden_states[-1][:, -1, :]

            if self.coeff_type == 'cosine':
                cnt = 0
                if version in [1,2]:
                    length = x_f['input_ids'].shape[0]
                else: 
                    length = x_r['input_ids'].shape[0]
                for idx in range(length):
                    temp_sum = 0
                    for j in range(k):
                        # 计算余弦相似度的dot product和norm
                        if version in [1,2]:
                            cos_similarity = torch.nn.functional.cosine_similarity(embeddings_f[idx].unsqueeze(0), embeddings_r[idx + cnt + j].unsqueeze(0))
                        elif version == 3:
                            cos_similarity = torch.nn.functional.cosine_similarity(embeddings_r[idx].unsqueeze(0), embeddings_f[idx + cnt + j].unsqueeze(0))
                        temp_sum += exp(abs(cos_similarity) / self.alpha)
                    cnt += 1
                    total_coeff.append(temp_sum)
                    

            elif self.coeff_type == 'distance':
                for idx in range(x_f['input_ids'].shape[0]):
                    temp_sum = 0
                    for j in range(k):
                        similarity = torch.cdist(embeddings_f[idx].unsqueeze(0).float(), embeddings_r[idx + j].unsqueeze(0).float(), p=2)
                        temp_sum += exp(similarity / self.alpha)
                    total_coeff.append(temp_sum)
            # 暂时先不对semantic_entropy进行处理
            elif self.coeff_type == 'semantic_entropy':
                from .semantic_entropy import EntailmentPythia
                for idx in range(x_f['input_ids'].shape[0]):
                    temp_sum = 0
                    for j in range(x_r['input_ids'].shape[0]):
                        input_ids = torch.cat((x_f['input_ids'][idx].unsqueeze(0),
                                               x_r['input_ids'][j].unsqueeze(0)),dim=0)
                        semantic_entropy = pythia.compute_semantic_entropy(input_ids=input_ids)
                        temp_sum += exp(semantic_entropy / self.alpha)

            if version == 1:
                for idx in range(x_f['input_ids'].shape[0]):
                    cnt = 0
                    log_ratio_f = outputs_f_ref.logits[idx] - outputs_f.logits[idx] - log(k)
                    
                    for j in range(k):
                        # coefficient calculation
                        if self.coeff_type == 'cosine':
                            # 计算余弦相似度
                            coeff = torch.nn.functional.cosine_similarity(embeddings_f[idx].unsqueeze(0), embeddings_r[idx + cnt + j].unsqueeze(0))
                        elif self.coeff_type == 'distance':
                            coeff = torch.cdist(embeddings_f[idx].unsqueeze(0).float(),
                                            embeddings_r[idx + j].unsqueeze(0).float(), p=2)
                            # temp1 = (exp((1 - coeff) / self.alpha) / total_coeff[idx]) * F.logsigmoid(log_ratio_1)
                        elif self.coeff_type == 'semantic_entropy':
                            return
                        
                        # loss calculation
                        log_ratio_r = outputs_r.logits[idx + cnt + j] + log(k) - outputs_r_ref.logits[idx + cnt + j] 
                        retain_loss = (k / (k + 1)) * F.logsigmoid(self.beta * log_ratio_r) * (exp(coeff / self.alpha) / total_coeff[idx])
                        forget_loss = (1 / (k + 1)) * F.logsigmoid(self.beta * log_ratio_f) 
                        loss += retain_loss + forget_loss
                    cnt += 1

            elif version == 2:
            # 更新loss计算，避免重复计算
                for idx in range(x_f['input_ids'].shape[0]):
                    cnt = 0
                    log_ratio_f = outputs_f.logits[idx] - outputs_f_ref.logits[idx] - log(k)
                    
                    for j in range(k):
                        # coefficient calculation
                        if self.coeff_type == 'cosine':
                            # 计算余弦相似度
                            coeff = torch.nn.functional.cosine_similarity(embeddings_f[idx].unsqueeze(0), embeddings_r[idx + cnt + j].unsqueeze(0))
                        elif self.coeff_type == 'distance':
                            coeff = torch.cdist(embeddings_f[idx].unsqueeze(0).float(),
                                            embeddings_r[idx + j].unsqueeze(0).float(), p=2)
                            # temp1 = (exp((1 - coeff) / self.alpha) / total_coeff[idx]) * F.logsigmoid(log_ratio_1)
                        elif self.coeff_type == 'semantic_entropy':
                            return
                        # loss calculation
                        log_ratio_r = outputs_r_ref.logits[idx + cnt + j]+ log(k) - outputs_r.logits[idx + cnt + j] 
                        retain_loss = (k / (k + 1)) * F.logsigmoid(self.beta * log_ratio_r)
                        forget_loss = (1 / (k + 1)) * F.logsigmoid(self.beta * log_ratio_f) * (exp(coeff / self.alpha) / total_coeff[idx])
                        loss += retain_loss + forget_loss
                    cnt += 1
            
            elif version == 3:
                for idx in range(x_r['input_ids'].shape[0]):
                    cnt = 0
                    log_ratio_r = outputs_r.logits[idx] - outputs_r_ref.logits[idx] - log(k)

                    for j in range(k):
                        # coefficient calculation
                        if self.coeff_type == 'cosine':
                            coeff = torch.nn.functional.cosine_similarity(embeddings_r[idx].unsqueeze(0), embeddings_f[idx + cnt + j].unsqueeze(0))
                        log_ratio_f = log(k) + outputs_f_ref.logits[idx + cnt + j] - outputs_f.logits[idx + cnt + j]
                        # 此处的系数为1-cos是因为希望在正负样本比较近的时候，给予负样本更大权重
                        retain_loss = (1 / (k + 1)) * (exp(abs(coeff) / self.alpha) / total_coeff[idx]) * F.logsigmoid(self.beta * log_ratio_r)
                        forget_loss = (k / (k + 1)) * F.logsigmoid(self.beta * log_ratio_f)
                        loss += retain_loss + forget_loss
                    cnt += 1

            if version == 1:
                loss = -(2 / self.beta) * loss.mean() / (k * x_f['input_ids'].shape[0])
            elif version == 2:
                loss = (2 / self.beta) * loss.mean() / (k * x_f['input_ids'].shape[0]) 
            elif version == 3:
                loss = -(2 / self.beta) * loss.mean() / (k * x_r['input_ids'].shape[0]) 

            wandb.log({"loss": loss.item()}) 
            if torch.isnan(loss):
                print("NaN detected in loss") 
                print("total coeff is: ", total_coeff)

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
