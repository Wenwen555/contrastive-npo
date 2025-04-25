import os
os.environ["USE_TENSOR_PARALLEL"] = "0"
os.environ["TORCH_DISABLE_DTENSOR"] = "1"

import sys
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import argparse
from datetime import timedelta  # 添加这行导入

sys.path.insert(0, os.path.abspath(os.path.join(__file__, "../../..")))
from baselines.baselines.dataset import PairedDataset
from baselines.baselines.utils import load_model_and_tokenizer

class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = nn.Conv2d(3, 3, (3, 3), 1, 1)
    
    def forward(self, x):
        return self.m(x)
    

def select_device(device="", batch_size=0, newline=True):
    """Selects computing device (CPU, CUDA GPU, MPS) for YOLOv5 model deployment, logging device info."""
    s = f"torch-{torch.__version__} "
    device = str(device).strip().lower().replace("cuda:", "").replace("none", "")  # to string, 'cuda:0' to '0'
    cpu = device == "cpu"
    mps = device == "mps"  # Apple Metal Performance Shaders (MPS)
    if cpu or mps:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # force torch.cuda.is_available() = False
    elif device:  # non-cpu device requested
        os.environ["CUDA_VISIBLE_DEVICES"] = device  # set environment variable - must be before assert is_available()
        assert torch.cuda.is_available() and torch.cuda.device_count() >= len(device.replace(",", "")), (
            f"Invalid CUDA '--device {device}' requested, use '--device cpu' or pass valid CUDA device(s)"
        )

    if not cpu and not mps and torch.cuda.is_available():  # prefer GPU if available
        devices = device.split(",") if device else "0"  # range(torch.cuda.device_count())  # i.e. 0,1,6,7
        n = len(devices)  # device count
        if n > 1 and batch_size > 0:  # check batch_size is divisible by device_count
            assert batch_size % n == 0, f"batch-size {batch_size} not multiple of GPU count {n}"
        space = " " * (len(s) + 1)
        for i, d in enumerate(devices):
            p = torch.cuda.get_device_properties(i)
            s += f"{'' if i == 0 else space}CUDA:{d} ({p.name}, {p.total_memory / (1 << 20):.0f}MiB)\n"  # bytes to MB
        arg = "cuda:0"
    elif mps and getattr(torch, "has_mps", False) and torch.backends.mps.is_available():  # prefer MPS if available
        s += "MPS\n"
        arg = "mps"
    else:  # revert to CPU
        s += "CPU\n"
        arg = "cpu"

    if not newline:
        s = s.rstrip()
    print(s)
    return torch.device(arg)


def setup(rank, world_size):
    # 明确设置环境变量
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '29500'  # 改用更常用的端口
    os.environ['OMP_NUM_THREADS'] = '1'  # 显式设置避免警告
    
    # 初始化进程组
    dist.init_process_group(
        backend="nccl",
        # init_method="env://",
        # rank=rank,
        # world_size=world_size,
        timeout=timedelta(seconds=3000) # 增加超时时间
    )


def cleanup():
    dist.destroy_process_group()

def train_epoch(model, dataloader, optimizer, scheduler, epoch, device, local_rank):
    model.train()
    total_loss = 0.0
    
    if local_rank == 0:
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    else:
        progress_bar = dataloader

    for batch in progress_bar:
        optimizer.zero_grad()
        
        # 确保数据在正确设备上
        input_ids = batch['input_ids'].to(device, non_blocking=True)
        attention_mask = batch['attention_mask'].to(device, non_blocking=True)
        labels = batch['labels'].to(device, non_blocking=True)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        if local_rank == 0:
            progress_bar.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)

def finetune(args):
    # 初始化分布式
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    devices = os.environ["CUDA_VISIBLE_DEVICES"]
    device = select_device(devices)
    torch.cuda.set_device(local_rank)
    print(local_rank)
    setup(local_rank, world_size)
    
    device = torch.device("cuda", local_rank)
    
    # 加载模型
    if local_rank == 0:
        print("Loading model...")
    model, tokenizer = load_model_and_tokenizer(
        args.model_dir,
        tokenizer_dir=args.tokenizer_dir
    )
    tokenizer.pad_token = tokenizer.eos_token

    test_model = TestModel()
    test_model = test_model.to(device)
    
    # 移动到设备并包装DDP
    model = model.to(device)
    model = DDP(test_model, device_ids=[local_rank], output_device=local_rank, static_graph=True)
    
    # 数据加载
    dataset = PairedDataset(
        args.data_file,
        tokenizer=tokenizer,[]
        max_len=args.max_len,
        mode='finetuning'
    )
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=local_rank,
        shuffle=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.per_device_batch_size,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        pin_memory_device=str(device)
    )
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs * len(dataloader)
    )
    
    # 训练
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)  # 确保每个epoch有不同的shuffle
        
        start_time = time.time()
        avg_loss = train_epoch(model, dataloader, optimizer, scheduler, epoch, device, local_rank)
        epoch_time = time.time() - start_time
        
        if local_rank == 0:
            print(f"Epoch {epoch} completed in {epoch_time:.2f}s - Loss: {avg_loss:.4f}")
    
    # 保存模型
    if local_rank == 0:
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
        
        model.module.save_pretrained(args.out_dir)
        tokenizer.save_pretrained(args.out_dir)
        print(f"Model saved to {args.out_dir}")
    
    cleanup()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--tokenizer_dir', type=str, default=None)
    parser.add_argument('--data_file', type=str, required=True)
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--max_len', type=int, default=4096)
    parser.add_argument('--per_device_batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=5)
    args = parser.parse_args()
    
    finetune(args)