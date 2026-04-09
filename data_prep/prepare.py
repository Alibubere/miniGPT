import os
import numpy as np
import logging
from tokenization import SimpleBPE


def prepare_data(
    file_path: str,
    output_dir: str,
    train_file_name: str,
    val_file_name: str,
    vocab_size: int = 256,
):
    """
    Tokenize a text file and save train/val splits as binary memmap files.

    If the output .bin files already exist, skips tokenization and training,
    and loads the tokenizer from saved vocab and merges JSON files instead.
    The text is split 90% train / 10% validation before tokenization.

    Args:
        file_path (str): Path to the input .txt file.
        output_dir (str): Directory where output files will be saved.
        train_file_name (str): Filename for the training binary (must end in .bin).
        val_file_name (str): Filename for the validation binary (must end in .bin).
        vocab_size (int): Target BPE vocabulary size. Defaults to 256.

    Returns:
        tuple: (tokenizer, train_path, val_path)
            - tokenizer (SimpleBPE): Trained or loaded tokenizer instance.
            - train_path (str): Full path to the training .bin file.
            - val_path (str): Full path to the validation .bin file.

    Raises:
        FileNotFoundError: If file_path does not exist.
        ValueError: If file_path is not a .txt file or output files are not .bin format.
    """

    if not os.path.exists(file_path):
        logging.error(f"file does not exist: {file_path}")
        raise FileNotFoundError(file_path)

    if not file_path.endswith(".txt"):
        raise ValueError("Input file must be a .txt file.")

    # Check if BOTH files have the correct extension
    if not (train_file_name.endswith(".bin") and val_file_name.endswith(".bin")):
        error_msg = f"Output files must be .bin format. Received: {train_file_name}, {val_file_name}"
        logging.error(error_msg)
        raise ValueError(error_msg)

    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, train_file_name)
    val_path = os.path.join(output_dir, val_file_name)

    vocab_path = os.path.join(output_dir, "vocab.json")
    merges_path = os.path.join(output_dir, "merges.json")

    if os.path.exists(train_path) and os.path.exists(val_path):
        logging.info(f"Data already exist {train_path} and {val_path}")
        tokenizer = SimpleBPE()
        tokenizer.load_vocab_and_merges(vocab_path, merges_path)
        return tokenizer, train_path, val_path

    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    n = len(text)

    train_text = text[: int(n * 0.9)]
    val_text = text[int(n * 0.9) :]

    tokenizer = SimpleBPE()
    logging.info(f"Training tokenizer with vocab_size={vocab_size}...")
    tokenizer.train(train_text, vocab_size)

    tokenizer.save_vocab_and_merges(vocab_path, merges_path)

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    dtype = np.uint16 if vocab_size <= 65535 else np.uint32

    train_ids_np = np.array(train_ids, dtype=dtype)
    val_ids_np = np.array(val_ids, dtype=dtype)

    train_ids_np.tofile(train_path)
    val_ids_np.tofile(val_path)
    logging.info(
        f"Success! Train size: {len(train_ids_np)}, Val size: {len(val_ids_np)}"
    )
    logging.info(f"Saved to {output_dir} using {dtype.__name__}")
    return tokenizer, train_path, val_path
