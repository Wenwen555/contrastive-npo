from .logger import RougeEvalLogger
from tqdm.contrib import tzip
from typing import List
import random

def get_prefix_before_words_occur(string: str, words: List[str]) -> str:
    for word in words: string = string.split(word)[0]
    return string


def eval(
    model, tokenizer,
    qa, 
    icl_qa,
    max_new_tokens : int = 100,
):

    logger = RougeEvalLogger()
    general_prompt: str = ""
    
    list_icl_qa = []
    for environment in icl_qa:
        for sample in environment.values():
            list_icl_qa.append(sample)


    # 假如是0 prompt,看看效果会不会变好or变坏
    # 0 prompt效果糟糕!
    random.shuffle(list_icl_qa)
    cnt = 0
    for sample in list_icl_qa:
        if cnt >= 2:
            break
        general_prompt += f"In the context: {sample['context']} \n"
        for output in sample['model_output']:
            general_prompt += f"Question: {output['question']}\nAnswer: {output['answer']}\n\n"
        cnt += 1

    for sample in qa:
        prompt = general_prompt + f"In the context: {sample['context']}\n\n"
        for output in sample['model_output']:
            question = output['question']
            answer = output['answer']
            prompt += f"Question: {question}\nAnswer: "
            # Encode the `prompt` into `input_ids`
            input_ids = tokenizer(
                prompt,
                return_tensors='pt',
                add_special_tokens=True).input_ids
            
            # Use the `model` to generate the continuation of the `input_ids`.
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
            # import ipdb
            # ipdb.set_trace()

            output = get_prefix_before_words_occur(output, ["\n\n", "\nQuestion", "Question:"])
            logger.log(prompt, answer, output, question=question)

    return logger.report()
