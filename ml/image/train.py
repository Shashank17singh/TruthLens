"""Train one image detector.

    python -m ml.image.train model=convnext_tiny use_frequency=true
    python -m ml.image.train -m model=efficientnet_b0,convnext_tiny,vit_b16 use_frequency=true,false
    python -m ml.image.train augment=none

Re-running the same command resumes from the last epoch, so a Colab session that
ends mid-training loses at most one epoch.
"""

import math
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from ml.common.manifest import read_manifest
from ml.common.seed import set_seed
from ml.common.tracking import Tracker
from ml.image.augment import eval_transform, train_transform
from ml.image.data import eval_loader, predict_logits, train_loader
from ml.image.model import build_model


def run_name(config: dict) -> str:
    branch = "freq" if config["use_frequency"] else "rgb"
    return f"image-{config['model']['name']}-{branch}-{config['augment']['name']}-{config['version']}"


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


def train(config: dict) -> Path:
    """Train until early stopping and return the path of the best checkpoint."""
    set_seed(config["seed"])
    device = resolve_device(config["device"])
    use_amp = config["amp"] and device == "cuda"
    name = run_name(config)
    directory = Path(config["checkpoint_dir"]) / name
    best_path = directory / f"{name}-best.pt"
    last_path = directory / f"{name}-last.pt"

    frame = read_manifest(Path(config["manifest"]))
    root = Path(config["data_root"])
    model = build_model(config).to(device)
    model_config = config["model"]

    loader = train_loader(
        frame[frame["split"] == "train"],
        root,
        train_transform(config["augment"], config["image_size"], model.mean, model.std, config["seed"]),
        model_config["batch_size"],
        config["num_workers"],
        config["seed"],
    )
    validation = frame[frame["split"] == "val"].reset_index(drop=True)
    validation_loader = eval_loader(
        validation, root, eval_transform(model.mean, model.std), model_config["batch_size"] * 2, config["num_workers"]
    )

    accumulation = model_config["accumulation_steps"]
    batches = min(len(loader), config["steps_per_epoch"] or len(loader))
    updates_per_epoch = max(1, batches // accumulation)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=model_config["learning_rate"], weight_decay=config["weight_decay"]
    )
    scheduler = warmup_cosine(
        optimizer, config["warmup_epochs"] * updates_per_epoch, config["epochs"] * updates_per_epoch
    )
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
    for epoch in range(start_epoch, config["epochs"]):
        if stale_epochs >= config["patience"]:
            break
        model.train()
        started = time.perf_counter()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, (images, labels) in enumerate(loader):
            if step >= batches:
                break
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.float32, non_blocking=True)
            with torch.autocast(device_type=device, enabled=use_amp):
                loss = criterion(model(images), smooth_targets(labels, config["label_smoothing"]))
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
        auc = float(roc_auc_score(validation["label"], logits))
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
            "mean": model.mean,
            "std": model.std,
        }
        temporary = last_path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(last_path)
        if improved:
            torch.save({key: state[key] for key in ("model", "epoch", "best_auc", "config", "name", "mean", "std")}, best_path)

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
        print(f"epoch {epoch}: loss {total_loss / batches:.4f}, val AUC {auc:.4f}, best {best_auc:.4f}")

    tracker.finish()
    return best_path


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(config: DictConfig) -> None:
    print(train(OmegaConf.to_container(config, resolve=True)))


if __name__ == "__main__":
    main()
