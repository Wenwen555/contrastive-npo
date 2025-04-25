from .logger import RougeEvalLogger

from tqdm.contrib import tzip
from typing import List


def eval(
    model, 
    tokenizer,
    data: str,
    max_new_tokens: int = 128,
    interactive: bool = True  # 新增交互模式开关
):
    logger = RougeEvalLogger()
    
    # ============= 交互式调试模式 =============
    if interactive:
        print("\n===== 进入交互调试模式 (输入 'exit' 退出) =====")
        while True:
            try:
                user_prompt = input("\n请输入测试prompt: ").strip()
                if user_prompt.lower() == 'exit':
                    break
                
                # 处理用户输入
                input_ids = tokenizer(
                    user_prompt,
                    return_tensors='pt',
                    add_special_tokens=True
                ).input_ids.to(model.device)
                
                # 生成输出
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id
                )
                
                # 解码并打印结果
                output = tokenizer.decode(
                    output_ids[0][len(input_ids[0]):], 
                    skip_special_tokens=True
                )
                print(f"\n模型输出: {output}")
                print("-" * 50)
                
            except Exception as e:
                print(f"错误: {str(e)}")
                continue
        return None  # 交互模式不返回评估报告
    # ============= 正常评估模式 =============
    
    for sample in data:
        prompt_list = sample['model_output']
        gt = sample['context']
        for propt in prompt_list:
            input_ids = tokenizer(
                propt,
                return_tensors='pt',
                add_special_tokens=True
            ).input_ids
            assert len(input_ids) == 1

            gt_ids = tokenizer(gt, return_tensors='pt', add_special_tokens=True).input_ids[:, :max_new_tokens]
            output_ids = model.generate(
                input_ids.to(model.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id)
            
            output_ids = output_ids[:, len(input_ids[0]):]
            output = tokenizer.batch_decode(
                output_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True)[0]
            gt_short = tokenizer.batch_decode(
                gt_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True)[0]
            logger.log(propt, gt_short, output)
    return logger.report()

    for prompt, gt in tzip(prompts, gts):
        # Encode the `prompt` into `input_ids`
        input_ids = tokenizer(
            prompt,
            return_tensors='pt',
            add_special_tokens=True
        ).input_ids
        assert len(input_ids) == 1

        gt_ids = tokenizer(gt, return_tensors='pt', add_special_tokens=True).input_ids[:, :max_new_tokens]

        # Use the `model` to generate the continuation of the `input_ids`.
        output_ids = model.generate(
            input_ids.to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id)
        output_ids = output_ids[:, len(input_ids[0]):] # size is (batch, len)
        output = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)[0]
        gt_short = tokenizer.batch_decode(
            gt_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True)[0]
        logger.log(prompt, gt_short, output)

    return logger.report()
