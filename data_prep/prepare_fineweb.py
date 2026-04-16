from datasets import load_dataset
import numpy as np
import os
import logging
from transformers import GPT2TokenizerFast


def prepare_fineweb(
    data_dir,
    vocab_size=50304,          # ignored now, GPT2 tokenizer has fixed 50257 vocab
    num_train_tokens=500_000_000,
    val_tokens_target=100_000,
):
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    eos_id = tokenizer.eos_token_id  # 50256

    if os.path.exists(train_path) and os.path.exists(val_path):
        logging.info("Binary data already exists. Skipping tokenization.")
        return tokenizer

    logging.info("Loading dataset stream...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    dtype = np.uint16  # GPT2 vocab (50257) fits in uint16
    CHUNK_SIZE = 1_000_000

    train_buffer = []
    val_buffer = []
    train_total = 0
    val_total = 0
    doc_count = 0
    val_done = False

    train_file = open(train_path, "wb")
    val_file = open(val_path, "wb")

    def flush(buffer, file):
        arr = np.array(buffer, dtype=dtype)
        arr.tofile(file)
        buffer.clear()

    logging.info("Tokenizing and writing binary files...")

    for example in dataset:
        ids = tokenizer.encode(example["text"], add_special_tokens=False)
        ids.append(eos_id)

        if not val_done:
            val_buffer.extend(ids)
            val_total += len(ids)
            if val_total >= val_tokens_target:
                val_done = True
                flush(val_buffer, val_file)
                logging.info(f"Val set done: {val_total:,} tokens")
        else:
            train_buffer.extend(ids)
            train_total += len(ids)

            if len(train_buffer) >= CHUNK_SIZE:
                flush(train_buffer, train_file)

        doc_count += 1

        if doc_count % 5000 == 0:
            logging.info(
                f"Docs: {doc_count:,} | Train tokens: {train_total:,} / {num_train_tokens:,}"
            )

        if train_total >= num_train_tokens:
            break

    if train_buffer:
        flush(train_buffer, train_file)
    if val_buffer:
        flush(val_buffer, val_file)

    train_file.close()
    val_file.close()

    logging.info(f"Done. Train: {train_total:,} tokens | Val: {val_total:,} tokens")
    return tokenizer