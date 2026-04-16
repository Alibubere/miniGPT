import torch
import logging
from torch import amp


def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    dataloader,
    device,
    epoch: int,
    scheduler=None,
    scaler=None,
    use_amp: bool = False,
    grad_accum_steps: int = 4,
):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()  # moved outside loop

    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        is_last_step = (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader)

        with amp.autocast(device_type="cuda", enabled=use_amp):
            logits, loss = model(X, y)
            loss = loss / grad_accum_steps  # normalize loss across accumulation steps

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if is_last_step:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

            optimizer.zero_grad()  

        running_loss += loss.item() * grad_accum_steps  # undo normalization for logging

        if batch_idx % 500 == 0:
            logging.info(
                f"Epoch [{epoch}] Batch [{batch_idx}/{len(dataloader)}] "
                f"loss: {loss.item() * grad_accum_steps:.4f}"
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
                    f"loss: {loss.item():.4f}"
                )

    val_loss /= len(dataloader)
    return val_loss