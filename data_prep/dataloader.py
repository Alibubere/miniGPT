import torch
from torch.utils.data import DataLoader


def get_dataloader(
    dataset,
    batch_size: int = 4,
    shuffle: bool = False,
    num_workers: int = 2,
    pin_memory: bool = False,
    drop_last: bool = False,
    persistent_workers=False,
    prefetch_factor=2,
):

    return DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )