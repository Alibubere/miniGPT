import torch
import os
from model_utils.train import train_one_epoch, validate_one_epoch
from model_utils.training_utils import save_checkpoint , load_checkpoint
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
):

    start_epoch = 1
    best_val_loss = float("inf")
    checkpoint_to_load = None

    if resume:
        latest_exists = os.path.exists(latest_path)
        best_exists = os.path.exists(best_path)

        if latest_exists and best_exists:

            latest_loss = torch.load(latest_path, map_location="cpu").get(
                "best_val_loss", float("inf")
            )
            best_loss = torch.load(best_path, map_location=device).get(
                "best_val_loss", float("inf")
            )

            if latest_loss <= best_loss:
                checkpoint_to_load = latest_path
                path_type = "LATEST (Better Performance)"
                best_val_loss = latest_loss
            else:
                checkpoint_to_load = best_path
                path_type = "BEST (Better Performance)"
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
                # Load the selected checkpoint completely using your utility function
                checkpoint = load_checkpoint(
                    model, optimizer, checkpoint_to_load, scheduler, scaler, device
                )

                # Note: start_epoch is set to the *next* epoch
                start_epoch = checkpoint.get("epoch", 0) + 1
                model = checkpoint["model"]
                optimizer = checkpoint["optimizer"]
                scheduler = checkpoint["scheduler"]
                scaler = checkpoint["scaler"]

                loaded_best_val_loss = checkpoint.get("best_val_loss")
                if (
                    loaded_best_val_loss is not None
                    and loaded_best_val_loss < best_val_loss
                ):
                    best_val_loss = loaded_best_val_loss

                logging.info(
                    f"Checkpoint loaded from {checkpoint_to_load} ({path_type}). Resuming training from epoch {start_epoch}, best loss tracked: {best_val_loss:.4f}"
                )
            except RuntimeError as e:
                if "state_dict" in str(e):
                    logging.warning(
                        f"Model architecture mismatch. Starting from scratch. Error: {e}"
                    )
                    start_epoch = 1
                    best_val_loss = 10.0
                else:
                    raise

        else:
            logging.info(
                f"Resume enabled but no valid checkpoint found. Starting from scratch."
            )

    else:
        logging.info(f"Resume disabled. Starting from scratch.")

    # --- End of Checkpoint Selection Logic ---

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
        )

        val_avg_loss = validate_one_epoch(
            model=model,
            dataloader=val_dataloader,
            device=device,
            epoch=epoch,
            use_amp=use_amp,
        )

        logging.info(
            f"Epoch: [{epoch}/{num_epochs}]"
            f"Train loss: {train_avg_loss:.4f} | Val loss: {val_avg_loss:.4f}"
        )

        if val_avg_loss < best_val_loss:
            logging.info(
                f"Validation loss improved from {best_val_loss:.4f} to {val_avg_loss:.4f}. Saving Best model."
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
