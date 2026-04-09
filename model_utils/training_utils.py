import logging
import torch
import math
import os

def save_checkpoint(
    path: str,
    model,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    scaler,
    best_val_loss: float = None,
):
    """
    Save the checkpoint (calls save_checkpoint after every epoch for better storing the parameters)

    Args:
        path (str): The file path where the checkpoint will be saved.
        model (torch.nn.Module): The model to save its state_dict.
        optimizer (torch.optim.Optimizer): The optimizer to save its state_dict.
        scheduler (optional): The LR scheduler. Its state_dict is saved if not None.
        epoch (int): The number of the *next* epoch to run (i.e., last_completed_epoch + 1).
        scaler (optional): The gradient scaler (for mixed precision). Its state_dict is saved if not None.
        best_val_loss (float, optional): The current best validation loss achieved.
    """
    if path is None:
        raise FileNotFoundError(f"Path not found {path} for save checkpoint")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "sched_state": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
    }

    try:
        torch.save(checkpoint, path)
        logging.info(f"Checkpoint saved to {path} (resumption epoch: {epoch})")
    except Exception as e:
        logging.error(f"Error saving checkpoint to {path}: {e}")
        raise


def load_checkpoint(
    model,
    optimizer: torch.optim.Optimizer,
    path: str,
    scheduler,
    scaler,
    device,
):
    """
    Loads a PyTorch checkpoint and restores model, optimizer, scheduler, and scaler state.

    Args:
        model (torch.nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer): The optimizer to restore state.
        path (str): Path to the checkpoint file (.pth or .pt).
        scheduler (optional): The LR scheduler to restore state.
        scaler (optional): The gradient scaler to restore state.
        device (str or torch.device): Device to map the checkpoint to ("cuda", "cpu", etc.).

    Returns:
        dict: A dictionary containing the restored components and meta info:
              - 'model': model with loaded weights.
              - 'optimizer': optimizer with restored state.
              - 'epoch': int, the next epoch number to start training from.
              - 'scheduler': scheduler with restored state.
              - 'scaler': scaler with restored state.
              - 'best_val_loss': float , best val loss the model have given
    """

    try:
        map_location = torch.device(device)
        checkpoint = torch.load(path, map_location=map_location)
    except FileNotFoundError:
        logging.error(f"Checkpoint file not found at {path}")
        raise
    except Exception as e:
        logging.error(f"Error loading checkpoint from {path}: {e}")
        raise

    model.load_state_dict(checkpoint["model_state"])

    optimizer.load_state_dict(checkpoint["optimizer_state"])

    epoch = checkpoint.get("epoch", 1)
    best_val_loss = checkpoint.get("best_val_loss", None)

    scheduler_restored = False
    if scheduler is not None and checkpoint.get("sched_state") is not None:
        try:
            scheduler.load_state_dict(checkpoint["sched_state"])
            scheduler_restored = True
        except Exception as e:
            logging.warning(f"Could not load scheduler state_dict: {e}")

    scaler_restored = False
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        try:
            scaler.load_state_dict(checkpoint["scaler_state"])
            scaler_restored = True
        except Exception as e:
            logging.warning(f"Could not load scaler state_dict: {e}")

    logging.info(
        f"Checkpoint loaded from {path}. Resuming at epoch {epoch}. (Scheduler restored: {scheduler_restored}, Scaler restored: {scaler_restored})"
    )

    return {
        "model": model,
        "optimizer": optimizer,
        "epoch": epoch,
        "scheduler": scheduler,
        "scaler": scaler,
        "best_val_loss": best_val_loss,
    }


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.1,  # min LR = max_lr * 0.1, don't decay to zero
):
    def lr_lambda(current_step: int):
        # Linear warmup
        if current_step < num_warmup_steps:
            return current_step / max(1, num_warmup_steps)

        # Cosine decay
        progress = (current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Don't decay all the way to zero
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
