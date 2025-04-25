import os
import json
import random
import pickle

def split_pii_data(data_path, f_qa_data_path, r_qa_data_path, forget_scal):
    """
    分割PII数据为retain1, retain2和forget三个子集
    """
    # scal取值0~100，表示遗忘数据比例
    # 创建输出目录
    output_dir = f"data/pii/scal-{forget_scal}"
    os.makedirs(output_dir, exist_ok=True)
    raw_output_dir = os.path.join(output_dir, 'raw')
    qa_out_dir = os.path.join(output_dir, 'knowmem')

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    with open(f_qa_data_path, 'rb') as f:
        f_qa_data = pickle.load(f)
    
    with open(r_qa_data_path, 'rb') as f:
        r_qa_data = pickle.load(f)
    
    # 注意forget_sample是同时包含了要遗忘的数据和retain数据的，所以最高层的数据分类发生在forget_scal上
    # 一个scal内有 raw 和 know，如此才能保障unlearn的数据和其QA对应
    forget_sample = {}
    # 那么此处的qa对选取就是针对选择的forget_sample而选的，不能任意选
    forget_qa = []
    # 请重点思考：当前并没有retain_set的qa，是否要重新生成呢？ 
    # 由于没有配对需求，直接在retain的context上生成即可(修改qa-generation.py逻辑即可)
    retain_qa = []

    keys = list(data.keys())
    shuffled_keys = keys.copy() 
    random.shuffle(shuffled_keys)
    forget_length = int((forget_scal/100) * len(data))
    if forget_length < 1:
        raise Warning("There is not sample needed to unlearn!!!")
    # data中的每一个sample是一个dict
    for key in shuffled_keys[:forget_length]:
        # 遗忘数据加载
        forget_sample[key] = data[key]
        # forget_qa数据加载
        forget_qa.append(f_qa_data[key])
        # retain_qa数据加载
        for idx in r_qa_data[key]:
            retain_qa.append(r_qa_data[key][idx])
    
    datasets = {
        'forget_sample': {
            'data': forget_sample,  # 你的数据变量
            'output_dir': raw_output_dir,
            'subdir': ''  
        },
        'forget_qa': {
            'data': forget_qa,
            'output_dir': qa_out_dir,
            'subdir': 'knowmem_f'  # 子目录名
        },
        'retain_qa': {
            'data': retain_qa,
            'output_dir': qa_out_dir,
            'subdir': 'knowmem_r'
        }
    }
    # 统一保存逻辑
    for name, config in datasets.items():
        # 构建完整路径
        save_dir = os.path.join(config['output_dir'], config['subdir'])
        os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
        
        # 保存JSON文件
        json_path = os.path.join(save_dir, f"{name}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config['data'], f) 
    return 


# 使用示例
if __name__ == "__main__":
    LOCAL_DATASET_DIR_PII = '/root/cnpo/data/pii-data/train_synthetic_replacements-1k.pkl'
    LOCAL_DATASET_DIR_PII_FORGET_QA = '/root/cnpo/data/pii-data/qa_pairs_1k.pkl'
    LOCAL_DATASET_DIR_PII_RETAIN_QA = '/root/cnpo/data/pii-data/retain_qa_pairs_4k.pkl'
    for scal in [5, 10, 20, 30, 40]:
        split_pii_data(LOCAL_DATASET_DIR_PII,LOCAL_DATASET_DIR_PII_FORGET_QA,LOCAL_DATASET_DIR_PII_RETAIN_QA, scal)