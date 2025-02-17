import os
from datasets import load_dataset
from utils import write_json, write_text

os.makedirs('data', exist_ok=True)
LOCAL_DATASET_DIR = './local_datasets'


for corpus, Corpus in zip(['news', 'books'], ['News', 'Books']):
    corpus_path = os.path.join(LOCAL_DATASET_DIR, f"MUSE-{Corpus}")

    for split in ['forget_qa', 'retain_qa', 'forget_qa_icl', 'retain_qa_icl']:
        data = load_dataset(corpus_path,'knowmem', split=split)
        questions, answers = data['question'], data['answer']
        knowmem = [
            {'question': question, 'answer': answer}
            for question, answer in zip(questions, answers)
        ]
        write_json(knowmem, f"data/{corpus}/knowmem/{split}.json")

    for split in ['forget']:
        data = load_dataset(corpus_path, 'verbmem', split='forget')
        prompts, gts = data['prompt'], data['gt']
        verbmem = [
            {'prompt': prompt, 'gt': gt}
            for prompt, gt in zip(prompts, gts)
        ]
        write_json(verbmem, f"data/{corpus}/verbmem/forget.json")

    for split in ['forget', 'retain', 'holdout']:
        privleak = load_dataset(corpus_path, 'privleak', split=split)['text']
        write_json(privleak, f"data/{corpus}/privleak/{split}.json")

    for split in ['forget', 'holdout', 'retain1', 'retain2']:
        raw = load_dataset(corpus_path, 'raw', split=split)['text']
        write_json(raw, f"data/{corpus}/raw/{split}.json")
        write_text("\n\n".join(raw), f"data/{corpus}/raw/{split}.txt")


corpus_path_news = os.path.join(LOCAL_DATASET_DIR, "MUSE-News")
for crit in ['scal', 'sust']:
    for fold in range(1, 5):
        data = load_dataset(corpus_path_news, crit, split=f"forget_{fold}")['text']
        write_json(data, f"data/news/{crit}/forget_{fold}.json")
        write_text("\n\n".join(data), f"data/news/{crit}/forget_{fold}.txt")


# import os
# from datasets import load_dataset
# from utils import write_json, write_text
#
# # 假设本地数据集存储在 ./local_datasets
# LOCAL_DATASET_DIR = './local_datasets'
#
# os.makedirs('data', exist_ok=True)
#
# for corpus, Corpus in zip(['news', 'books'], ['News', 'Books']):
#     corpus_path = os.path.join(LOCAL_DATASET_DIR, f"MUSE-{Corpus}")
#
#     for split in ['forget_qa', 'retain_qa', 'forget_qa_icl', 'retain_qa_icl']:
#         data = load_dataset(os.path.join(corpus_path,'knowmem'),split=split)
#         questions, answers = data['question'], data['answer']
#         knowmem = [
#             {'question': question, 'answer': answer}
#             for question, answer in zip(questions, answers)
#         ]
#         write_json(knowmem, f"data/{corpus}/knowmem/{split}.json")
#
#     for split in ['forget']:
#         data = load_dataset(corpus_path, 'verbmem', split='forget')
#         prompts, gts = data['prompt'], data['gt']
#         verbmem = [
#             {'prompt': prompt, 'gt': gt}
#             for prompt, gt in zip(prompts, gts)
#         ]
#         write_json(verbmem, f"data/{corpus}/verbmem/forget.json")
#
#     for split in ['forget', 'retain', 'holdout']:
#         privleak = load_dataset(corpus_path, 'privleak', split=split)['text']
#         write_json(privleak, f"data/{corpus}/privleak/{split}.json")
#
#     for split in ['forget', 'holdout', 'retain1', 'retain2']:
#         raw = load_dataset(corpus_path, 'raw', split=split)['text']
#         write_json(raw, f"data/{corpus}/raw/{split}.json")
#         write_text("\n\n".join(raw), f"data/{corpus}/raw/{split}.txt")
#
# for crit in ['scal', 'sust']:
#     for fold in range(1, 5):
#         crit_path = os.path.join(LOCAL_DATASET_DIR, "MUSE-News")
#         data = load_dataset(crit_path, crit, split=f"forget_{fold}")['text']
#         write_json(data, f"data/news/{crit}/forget_{fold}.json")
#         write_text("\n\n".join(data), f"data/news/{crit}/forget_{fold}.txt")