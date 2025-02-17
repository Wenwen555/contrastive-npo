from .dataset import DefaultDataset
from .utils import load_model_and_tokenizer
import os

os.environ["CUDA_VISIBLE_DEVICES"]= "1,2"
import torch
print(f"可见的 GPU 数量: {torch.cuda.device_count()}")
print(f"当前使用的 GPU 索引: {torch.cuda.current_device()}")

import transformers
import argparse

def finetune(
    model_dir: str,
    data_file: str,
    out_dir: str,
    epochs: int = 5,
    per_device_batch_size: int = 2,
    learning_rate: float = 1e-5,
    max_len: int = 4096,
    tokenizer_dir: str | None = None
):
    model, tokenizer = load_model_and_tokenizer(
        model_dir,
        tokenizer_dir=tokenizer_dir
    )

    dataset = DefaultDataset(
        data_file,
        tokenizer=tokenizer,
        max_len=max_len
    )

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        optim='adamw_torch',
        lr_scheduler_type='cosine',
        bf16=True,
        report_to='none'        # Disable wandb
    )

    trainer = transformers.Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=dataset.get_collate_fn()
    )

    model.config.use_cache = False  # silence the warnings.
    trainer.train()
    trainer.save_model(out_dir)


def main():
    args = get_args()
    finetune(
        model_dir=args.model_dir,
        tokenizer_dir=args.model_dir,
        data_file=args.data_file,
        out_dir=args.out_dir,
        epochs=args.epochs,
        per_device_batch_size=args.per_device_batch_size,
        learning_rate=args.lr,
        max_len=args.max_len,
    )
    return;


def get_args():
    parser = argparse.ArgumentParser(description="finetuning baselines")

    parser.add_argument(
        '--model_dir', type=str,
        help="Path to the target model's hf directory."
    )
    parser.add_argument(
        '--tokenizer_dir', type=str, default=None,
        help="Path to the tokenizer's hf directory. Defaults to the target model's directory."
    )
    parser.add_argument(
        '--data_file', type=str,
        help="Path to the forget set file."
    )
    parser.add_argument(
        '--out_dir', type=str,
        help="Path to the output model's hf directory. Creates the directory if it doesn't already exist."
    )

    parser.add_argument(
        '--max_len', type=int, default=4096,
        help="max length of input ids fed to the model"
    )

    # Gradient ascent & Gradient difference
    parser.add_argument('--per_device_batch_size', type=int, default=2)

    parser.add_argument(
        '--lr', type=float, default=1e-5,
        help="Learning rate if algo is either gradient ascent (ga), gradient difference (gd), or task vector (tv)."
    )
    parser.add_argument(
        '--epochs', type=int, default=5,
        help="Number of epochs of training if algo is either gradient ascent (ga), gradient difference (gd), or task vector (tv)."
    )

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()