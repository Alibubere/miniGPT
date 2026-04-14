# 🤖 miniGPT

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-Enabled-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![HuggingFace](https://img.shields.io/badge/Dataset-FineWeb--Edu-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)]()
[![Model](https://img.shields.io/badge/Architecture-GPT--2%20Style-blueviolet?style=for-the-badge)]()
[![BPE](https://img.shields.io/badge/Tokenizer-Custom%20BPE-orange?style=for-the-badge)]()

A GPT-2 style language model built from scratch in PyTorch, trained on the [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset with a custom Byte Pair Encoding tokenizer.

---

## ✨ Features

- **GPT-2 style architecture** — Causal self-attention, MLP blocks, LayerNorm with optional bias, weight tying
- **Custom BPE tokenizer** — Built from scratch with `SimpleBPE`, trained on 10k sampled documents
- **FineWeb-Edu dataset** — Streams and tokenizes 500M tokens for training
- **Mixed precision training** — AMP with `torch.amp.GradScaler` for faster GPU training
- **Cosine LR schedule** — With linear warmup (5% of total steps)
- **Checkpoint system** — Saves `latest.pth` and `best.pth`, supports resuming training
- **`torch.compile`** — Uses `aot_eager` backend for optimized execution

---

## 🏗️ Project Structure

```
miniGPT/
├── data_prep/
│   ├── prepare_fineweb.py   # Downloads, tokenizes & writes binary train/val data
│   ├── dataset.py           # MemmapDataset for memory-mapped binary files
│   ├── dataloader.py        # DataLoader factory
│   └── prepare.py
├── model_utils/
│   ├── train.py             # train_one_epoch & validate_one_epoch
│   ├── train_loop.py        # Full training loop with checkpoint resume logic
│   └── training_utils.py    # Cosine scheduler, save/load checkpoint helpers
├── model.py                 # GPTConfig, GPT, Block, CausalSelfAttention, MLP
├── tokenization.py          # SimpleBPE tokenizer (train, encode, decode, save, load)
├── miniGPT.py               # Main entry point
├── miniGPT.ipynb            # Notebook version
├── requirements.txt
└── data/                    # Generated: train.bin, val.bin, vocab.json, merges.json
```

---

## ⚙️ Model Configuration

| Hyperparameter | Value |
|---|---|
| Embedding size (`n_embd`) | 768 |
| Attention heads (`n_head`) | 6 |
| Layers (`n_layer`) | 6 |
| Context length (`block_size`) | 1024 |
| Vocabulary size | 8,000 |
| Dropout | 0.2 |
| Batch size | 32 |
| Learning rate | 3e-4 |
| Weight decay | 0.005 |
| Optimizer | AdamW (β₁=0.9, β₂=0.95) |
| Epochs | 3 |
| Train tokens | 500M |

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2. Run training

```bash
python miniGPT.py
```

On first run, the script will:
1. Stream FineWeb-Edu and train the BPE tokenizer on 10k documents
2. Tokenize 500M training tokens and write `data/train.bin` + `data/val.bin`
3. Train the GPT model, saving checkpoints to `models/`

Subsequent runs with `resume = True` will automatically load the best available checkpoint.

---

## 📦 Requirements

```
torch >= 2.0
datasets == 2.18.0
pyarrow == 12.0.1
datatrove
huggingface_hub
transformers
tokenizers
numpy
```

---

## 📁 Generated Files

| File | Description |
|---|---|
| `data/train.bin` | Tokenized training data (uint16 memmap) |
| `data/val.bin` | Tokenized validation data (uint16 memmap) |
| `data/vocab.json` | BPE vocabulary |
| `data/merges.json` | BPE merge rules |
| `models/latest.pth` | Most recent checkpoint |
| `models/best.pth` | Best validation loss checkpoint |
| `logs/gpt.log` | Training logs |

---

## 👤 Author

**Mohammad Ali Bubere**

[![GitHub](https://img.shields.io/badge/GitHub-Alibubere-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Alibubere)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Mohammad%20Ali%20Bubere-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mohammad-ali-bubere-a6b830384/)
[![Gmail](https://img.shields.io/badge/Gmail-alibubere989@gmail.com-EA4335?style=for-the-badge&logo=gmail&logoColor=white)](mailto:alibubere989@gmail.com)

---

## 📄 License

This project is licensed under the MIT License.
