from datasets import load_dataset
import numpy as np
import os
import logging
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tokenization import SimpleBPE


def train_tokenizer_on_sample(
    dataset,
    vocab_size=8000,          
    sample_docs=10_000,       # use 10k docs to train the tokenizer
    vocab_path="data/vocab.json",
    merges_path="data/merges.json",
):
    """Sample docs from the stream, train BPE, save and return tokenizer."""

    if os.path.exists(vocab_path) and os.path.exists(merges_path):
        logging.info("Found existing tokenizer. Loading...")
        tokenizer = SimpleBPE()
        tokenizer.load_vocab_and_merges(vocab_path, merges_path)
        return tokenizer

    logging.info(f"Collecting {sample_docs:,} docs to train tokenizer...")
    sample_text = []
    for i, example in enumerate(dataset):
        sample_text.append(example["text"])
        if i >= sample_docs:
            break

    combined = "\n".join(sample_text)
    logging.info(f"Tokenizer training text: {len(combined):,} chars")

    tokenizer = SimpleBPE()
    tokenizer.train(combined, vocab_size=vocab_size)
    tokenizer.save_vocab_and_merges(vocab_path, merges_path)
    logging.info(f"Tokenizer trained. Vocab size: {len(tokenizer.vocab)}")

    return tokenizer


def prepare_fineweb(
    data_dir,
    vocab_size=8000,
    num_train_tokens=500_000_000,
    val_tokens_target=100_000,
):
    os.makedirs(data_dir, exist_ok=True)

    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    vocab_path = os.path.join(data_dir, "vocab.json")
    merges_path = os.path.join(data_dir, "merges.json")

    logging.info("Loading dataset stream for tokenizer training...")
    dataset_for_tokenizer = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    tokenizer = train_tokenizer_on_sample(
        dataset=dataset_for_tokenizer,
        vocab_size=vocab_size,
        vocab_path=vocab_path,
        merges_path=merges_path,
    )

    if os.path.exists(train_path) and os.path.exists(val_path):
        logging.info("Binary data already exists. Skipping tokenization.")
        return tokenizer

    logging.info("Loading fresh dataset stream for data preparation...")
    dataset_for_data = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    dtype = np.uint16 if vocab_size <= 65535 else np.uint32
    CHUNK_SIZE = 1000000

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

    for example in dataset_for_data:
        try:
            ids = tokenizer.encode(example["text"])
        except KeyError:
            doc_count += 1
            continue

        if len(ids) == 0:
            doc_count += 1
            continue

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