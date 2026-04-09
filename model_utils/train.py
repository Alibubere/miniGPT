import torch
import logging
from torch import amp
import os


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader,
    device,
    epoch: int,
    scheduler=None,
    scaler=None,
    use_amp: bool = False,
):

    model.train()

    running_loss = 0.0

    for batch_idx, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        with amp.autocast(device_type="cuda", enabled=use_amp):
            logits, loss = model(X, y)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()

        if batch_idx % 900 == 0:
            logging.info(
                f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                f"loss: {loss.item():.4f} "
            )

    running_loss /= len(dataloader)

    return running_loss


def validate_one_epoch(
    model: torch.nn.Module, dataloader, device, epoch: int, use_amp: bool = False
):

    model.eval()

    val_loss = 0.0

    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(dataloader):

            X, y = X.to(device), y.to(device)

            with amp.autocast(device_type="cuda", enabled=use_amp):
                logits, loss = model(X, y)

            val_loss += loss.item()

            if batch_idx % 300 == 0:
                logging.info(
                    f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                    f"loss: {loss.item():.4f} "
                )
    val_loss /= len(dataloader)

    return val_loss
