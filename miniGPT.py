import logging
import os
import numpy as np
import torch
from data_prep.prepare import prepare_data
from data_prep.dataset import MemmapDataset
from data_prep.dataloader import get_dataloader
from model import GPT, GPTConfig

# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
vocab_size = 1000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# files
file_name = "input.txt"
data_dir = "data"
train_file_name = "train.bin"
val_file_name = "val.bin"


def log_setup():
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    file_name = "gpt.log"

    full_path = os.path.join(log_dir, file_name)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s %(message)s",
        handlers=[logging.FileHandler(full_path), logging.StreamHandler()],
    )
    logging.info("Logging initialize successfully")


def main():

    log_setup()
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32
    tokenizer, train_path, val_path = prepare_data(
        file_name,
        data_dir,
        train_file_name=train_file_name,
        val_file_name=val_file_name,
        vocab_size=vocab_size,
    )
    train_dataset = MemmapDataset(
        data_dir=data_dir, block_size=block_size, split="train", dtype=dtype
    )
    val_dataset = MemmapDataset(
        data_dir=data_dir, block_size=block_size, split="val", dtype=dtype
    )

    train_loader = get_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    val_loader = get_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
        drop_last=False,
        persistent_workers=True,
        prefetch_factor=2,
    )

    gptconfig = GPTConfig(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
    )
    model = GPT(gptconfig)

    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logging.info(f"Training on: {device}")
    logging.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")


if __name__ == "__main__":
    main()
