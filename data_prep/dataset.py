import torch
import os
import numpy as np
from torch.utils.data import Dataset


class MemmapDataset(Dataset):
    def __init__(self, data_dir, block_size, split="train"):
        super().__init__()

        filename = "train.bin" if split == "train" else "val.bin"
        path = os.path.join(data_dir, filename)

        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx : idx + self.block_size].astype(np.int64))

        y = torch.from_numpy(
            self.data[idx + 1 : idx + 1 + self.block_size].astype(np.int64)
        )

        return x, y
