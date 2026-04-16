import torch
import os
from model_utils.train import train_one_epoch, validate_one_epoch
from model_utils.training_utils import save_checkpoint, load_checkpoint
import logging


def train_loop(
    resume: bool,
    num_epochs: int,
    latest_path: str,
    best_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    device,
    train_dataloader,
    val_dataloader,
    use_amp: bool,
    grad_accum_steps: int = 4,  
):
    start_epoch = 1
    best_val_loss = float("inf")
    checkpoint_to_load = None

    if resume:
        latest_exists = os.path.exists(latest_path)
        best_exists = os.path.exists(best_path)

        if latest_exists and best_exists:
            latest_ckpt = torch.load(latest_path, map_location="cpu")
            best_ckpt = torch.load(best_path, map_location="cpu")

            latest_loss = latest_ckpt.get("best_val_loss", float("inf"))
            best_loss = best_ckpt.get("best_val_loss", float("inf"))

            if latest_loss <= best_loss:
                checkpoint_to_load = latest_path
                path_type = "LATEST"
                best_val_loss = latest_loss
            else:
                checkpoint_to_load = best_path
                path_type = "BEST"
                best_val_loss = best_loss

        elif latest_exists:
            checkpoint_to_load = latest_path
            path_type = "LATEST (Only Available)"
            ckpt_data = torch.load(latest_path, map_location="cpu")
            best_val_loss = ckpt_data.get("best_val_loss", best_val_loss)

        elif best_exists:
            checkpoint_to_load = best_path
            path_type = "BEST (Only Available)"
            ckpt_data = torch.load(best_path, map_location="cpu")
            best_val_loss = ckpt_data.get("best_val_loss", best_val_loss)

        if checkpoint_to_load:
            try:
                checkpoint = load_checkpoint(
                    model, optimizer, checkpoint_to_load, scheduler, scaler, device
                )
                start_epoch = checkpoint.get("epoch", 0) + 1
                model = checkpoint["model"]
                optimizer = checkpoint["optimizer"]
                scheduler = checkpoint["scheduler"]
                scaler = checkpoint["scaler"]

                loaded_best = checkpoint.get("best_val_loss")
                if loaded_best is not None and loaded_best < best_val_loss:
                    best_val_loss = loaded_best

                logging.info(
                    f"Loaded {path_type} checkpoint. "
                    f"Resuming from epoch {start_epoch}, best loss: {best_val_loss:.4f}"
                )
            except RuntimeError as e:
                if "state_dict" in str(e):
                    logging.warning(f"Architecture mismatch. Starting from scratch. Error: {e}")
                    start_epoch = 1
                    best_val_loss = float("inf")
                else:
                    raise
        else:
            logging.info("Resume enabled but no checkpoint found. Starting from scratch.")
    else:
        logging.info("Resume disabled. Starting from scratch.")

    logging.info(f"Gradient accumulation steps: {grad_accum_steps}")
    logging.info(f"Effective batch size: {train_dataloader.batch_size * grad_accum_steps}")

    for epoch in range(start_epoch, num_epochs + 1):

        train_avg_loss = train_one_epoch(
            model,
            optimizer,
            dataloader=train_dataloader,
            epoch=epoch,
            device=device,
            scheduler=scheduler,
            scaler=scaler,
            use_amp=use_amp,
            grad_accum_steps=grad_accum_steps,
        )

        val_avg_loss = validate_one_epoch(
            model=model,
            dataloader=val_dataloader,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
        )

        logging.info(
            f"Epoch [{epoch}/{num_epochs}] | "
            f"Train loss: {train_avg_loss:.4f} | Val loss: {val_avg_loss:.4f}"
        )

        if val_avg_loss < best_val_loss:
            logging.info(
                f"Val loss improved {best_val_loss:.4f} → {val_avg_loss:.4f}. Saving best model."
            )
            best_val_loss = val_avg_loss
            save_checkpoint(
                path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                scaler=scaler,
                best_val_loss=best_val_loss,
            )

        save_checkpoint(
            path=latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            scaler=scaler,
            best_val_loss=best_val_loss,
        )