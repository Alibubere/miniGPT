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
    """
    Construct a PyTorch DataLoader from a given dataset.

    Handles edge cases where persistent_workers and prefetch_factor
    are incompatible with num_workers=0 by disabling them automatically.

    Args:
        dataset (Dataset): The dataset to load from.
        batch_size (int): Number of samples per batch. Defaults to 4.
        shuffle (bool): Whether to shuffle data each epoch.
                        Set True for training, False for validation. Defaults to False.
        num_workers (int): Number of subprocesses for data loading. Defaults to 2.
        pin_memory (bool): If True, pins memory for faster GPU transfer.
                           Set True when training on CUDA. Defaults to False.
        drop_last (bool): If True, drops the last incomplete batch. Defaults to False.
        persistent_workers (bool): Keep worker processes alive between epochs.
                                   Ignored when num_workers=0. Defaults to False.
        prefetch_factor (int): Batches to prefetch per worker.
                               Ignored when num_workers=0. Defaults to 2.

    Returns:
        DataLoader: Configured PyTorch DataLoader instance.
    """

    return DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor,
    )
