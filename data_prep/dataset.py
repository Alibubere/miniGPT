import torch
import os
import numpy as np
from torch.utils.data import Dataset


class MemmapDataset(Dataset):
    """
    PyTorch Dataset that reads tokenized data from a memory-mapped binary file.

    Uses numpy memmap for efficient access to large datasets without
    loading the entire file into RAM. Each sample is a (input, target)
    pair of token ID sequences offset by one position for language modeling.

    Attributes:
        data (np.memmap): Memory-mapped array of token IDs.
        block_size (int): Length of each input sequence.
    """

    def __init__(
        self, data_dir: str, block_size: int, split="train", dtype=np.uint16
    ) -> None:
        """
        Args:
            data_dir (str): Directory containing train.bin and val.bin files.
            block_size (int): Number of tokens per sequence.
            split (str): Either 'train' or 'val'. Defaults to 'train'.
            dtype: Numpy dtype for reading the binary file.
                   Use uint16 for vocab_size <= 65535, uint32 otherwise.
                   Defaults to np.uint16.
        """

        super().__init__()

        filename = "train.bin" if split == "train" else "val.bin"
        path = os.path.join(data_dir, filename)

        self.data = np.memmap(path, dtype=dtype, mode="r")
        self.block_size = block_size

    def __len__(self):
        """
        Returns the number of valid sequences in the dataset.

        Returns:
            int: Total samples = len(data) - block_size.
        """

        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        """
        Retrieve an (input, target) pair at the given index.

        Args:
            idx (int): Index of the sequence.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - x: Input token IDs of shape (block_size,).
                - y: Target token IDs of shape (block_size,), shifted by 1.
        """

        x = torch.from_numpy(self.data[idx : idx + self.block_size].astype(np.int64))

        y = torch.from_numpy(
            self.data[idx + 1 : idx + 1 + self.block_size].astype(np.int64)
        )

        return x, y
