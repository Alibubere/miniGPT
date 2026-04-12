import logging
import os
import numpy as np
import torch
from data_prep.prepare_fineweb import prepare_fineweb
from data_prep.dataset import MemmapDataset
from data_prep.dataloader import get_dataloader
from model import GPT, GPTConfig
from model_utils.train_loop import train_loop
from model_utils.training_utils import get_cosine_schedule_with_warmup
import torch._dynamo
torch._dynamo.config.suppress_errors = True

# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 1024  # what is the maximum context length for predictions?
max_iters = 5000
vocab_size = 50304
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 786
n_head = 6
n_layer = 6
dropout = 0.2
resume = True
num_epochs = 50
weight_decay = 0.005

# files
file_name = "input.txt"
data_dir = "data"
train_file_name = "train.bin"
val_file_name = "val.bin"
model_dir = "models"
os.makedirs(model_dir, exist_ok=True)
latest_path = os.path.join(model_dir, "latest.pth")
best_path = os.path.join(model_dir, "best.pth")


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
    tokenizer = prepare_fineweb(data_dir)
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
    model = GPT(gptconfig).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )
    num_training_steps = num_epochs * len(train_loader)
    num_warmup_steps = int(0.05 * num_training_steps)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    scaler = torch.amp.GradScaler(device="cuda")

    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logging.info(f"Training on: {device}")
    logging.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    model = torch.compile(model=model,backend="aot_eager")
    train_loop(
        resume=resume,
        num_epochs=num_epochs,
        latest_path=latest_path,
        best_path=best_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        device=device,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        use_amp=True,
    )


if __name__ == "__main__":
    main()
