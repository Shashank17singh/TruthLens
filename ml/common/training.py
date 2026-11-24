"""Training loop shared by every modality.

AdamW with warmup and a cosine schedule, label smoothing, mixed precision,
gradient accumulation, early stopping on validation AUC, and resuming from
the last epoch, so a Colab session that ends mid-training loses at most one
epoch. Each modality builds its own model and loaders and hands them here.
"""

import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset, get_worker_info

from ml.common.balance import weighted_sampler
from ml.common.tracking import Tracker

BEST_KEYS = ("model", "epoch", "best_auc", "config", "name")


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def warmup_cosine(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, factor)


def smooth_targets(labels: torch.Tensor, amount: float) -> torch.Tensor:
    return labels * (1.0 - amount) + 0.5 * amount


def _seed_worker(worker_id: int) -> None:
    """Give each loader worker its own stream, or all would apply identical augmentations."""
    info = get_worker_info()
    seed = info.seed % 2**32
    np.random.seed(seed)
    if hasattr(info.dataset, "set_random_seed"):
        info.dataset.set_random_seed(seed)


def balanced_loader(dataset: Dataset, frame: pd.DataFrame, batch_size: int, workers: int, seed: int) -> DataLoader:
    """Training loader drawing labels and generators evenly; `frame` must be in the dataset's row order."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=weighted_sampler(frame.reset_index(drop=True), seed),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=workers > 0,
        drop_last=True,
    )


def ordered_loader(dataset: Dataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=torch.cuda.is_available())


@torch.inference_mode()
def predict_logits(model: nn.Module, loader: DataLoader, device: str, amp: bool) -> np.ndarray:
    model.eval()
    outputs = []
    for inputs, _ in loader:
        with torch.autocast(device_type=device, enabled=amp):
            outputs.append(model(inputs.to(device, non_blocking=True)).float().cpu())
    return torch.cat(outputs).numpy() if outputs else np.empty(0, dtype=np.float32)


def fit(
    model: nn.Module,
    loader: DataLoader,
    validation_loader: DataLoader,
    validation_labels: np.ndarray,
    config: dict,
    name: str,
    metadata: dict,
    before_epoch: Callable[[nn.Module, int], None] | None = None,
) -> Path:
    """Train until early stopping and return the best checkpoint's path.

    `metadata` is saved in every checkpoint, for whatever the modality needs to
    rebuild its preprocessing. `before_epoch` can change the model between
    epochs, for example to unfreeze a pretrained backbone.
    """
    device = resolve_device(config["device"])
    use_amp = config["amp"] and device == "cuda"
    directory = Path(config["checkpoint_dir"]) / name
    best_path = directory / f"{name}-best.pt"
    last_path = directory / f"{name}-last.pt"
    model = model.to(device)
    model_config = config["model"]

    accumulation = model_config["accumulation_steps"]
    batches = min(len(loader), config["steps_per_epoch"] or len(loader))
    updates_per_epoch = max(1, batches // accumulation)
    # Every parameter is registered now, so unfreezing later needs no new optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = warmup_cosine(optimizer, config["warmup_epochs"] * updates_per_epoch, config["epochs"] * updates_per_epoch)
    scaler = torch.amp.GradScaler(device, enabled=use_amp)
    criterion = nn.BCEWithLogitsLoss()

    # Below any real AUC, so the first epoch always produces a best checkpoint.
    start_epoch, best_auc, stale_epochs = 0, -1.0, 0
    if last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch, best_auc, stale_epochs = state["epoch"] + 1, state["best_auc"], state["stale_epochs"]
        print(f"resuming {name} at epoch {start_epoch}, best val AUC {best_auc:.4f}")

    tracker = Tracker(directory, name, config, config["tracker"], config["wandb_project"])
    budget = config["max_minutes"] and config["max_minutes"] * 60
    run_started, epoch_seconds = time.perf_counter(), 0.0
    for epoch in range(start_epoch, config["epochs"]):
        if stale_epochs >= config["patience"]:
            break
        # Stop before an epoch that would overrun the budget; last.pt lets the same command resume.
        if budget and epoch > start_epoch and time.perf_counter() - run_started + epoch_seconds > budget:
            print(f"time budget of {config['max_minutes']} minutes reached after epoch {epoch - 1}; rerun to resume")
            break
        if before_epoch is not None:
            before_epoch(model, epoch)
        model.train()
        started = time.perf_counter()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (inputs, labels) in enumerate(loader):
            if step >= batches:
                break
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.float32, non_blocking=True)
            with torch.autocast(device_type=device, enabled=use_amp):
                loss = criterion(model(inputs), smooth_targets(labels, config["label_smoothing"]))
            scaler.scale(loss / accumulation).backward()
            total_loss += loss.item()
            if (step + 1) % accumulation == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        logits = predict_logits(model, validation_loader, device, use_amp)
        auc = float(roc_auc_score(validation_labels, logits))
        improved = auc > best_auc
        best_auc, stale_epochs = (auc, 0) if improved else (best_auc, stale_epochs + 1)

        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_auc": best_auc,
            "stale_epochs": stale_epochs,
            "config": config,
            "name": name,
            **metadata,
        }
        temporary = last_path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(last_path)
        if improved:
            torch.save({key: state[key] for key in (*BEST_KEYS, *metadata)}, best_path)

        tracker.log(
            {
                "epoch": epoch,
                "train_loss": total_loss / batches,
                "val_auc": auc,
                "best_val_auc": best_auc,
                "learning_rate": scheduler.get_last_lr()[0],
                "epoch_seconds": time.perf_counter() - started,
            },
            step=(epoch + 1) * updates_per_epoch,
        )
        epoch_seconds = time.perf_counter() - started
        print(f"epoch {epoch}: loss {total_loss / batches:.4f}, val AUC {auc:.4f}, best {best_auc:.4f}")

    tracker.finish()
    return best_path
