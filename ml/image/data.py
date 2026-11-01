"""Manifest-backed image dataset and data loaders."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, get_worker_info

from ml.common.balance import weighted_sampler


class ManifestImages(Dataset):
    def __init__(self, frame: pd.DataFrame, root: Path, transform):
        self.frame = frame.reset_index(drop=True)
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, float]:
        row = self.frame.iloc[index]
        with Image.open(self.root / row["source_dataset"] / row["path"]) as image:
            pixels = np.asarray(image.convert("RGB"))
        return self.transform(image=pixels)["image"], float(row["label"])


def _seed_worker(worker_id: int) -> None:
    """Give each loader worker its own stream, or all would apply identical augmentations."""
    info = get_worker_info()
    seed = info.seed % 2**32
    np.random.seed(seed)
    if hasattr(info.dataset.transform, "set_random_seed"):
        info.dataset.transform.set_random_seed(seed)


def train_loader(frame: pd.DataFrame, root: Path, transform, batch_size: int, workers: int, seed: int) -> DataLoader:
    frame = frame.reset_index(drop=True)
    return DataLoader(
        ManifestImages(frame, root, transform),
        batch_size=batch_size,
        sampler=weighted_sampler(frame, seed),
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_seed_worker,
        generator=torch.Generator().manual_seed(seed),
        persistent_workers=workers > 0,
        drop_last=True,
    )


def eval_loader(frame: pd.DataFrame, root: Path, transform, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        ManifestImages(frame, root, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


@torch.inference_mode()
def predict_logits(model: torch.nn.Module, loader: DataLoader, device: str, amp: bool) -> np.ndarray:
    model.eval()
    outputs = []
    for images, _ in loader:
        with torch.autocast(device_type=device, enabled=amp):
            outputs.append(model(images.to(device, non_blocking=True)).float().cpu())
    return torch.cat(outputs).numpy() if outputs else np.empty(0, dtype=np.float32)
