"""Manifest-backed image dataset and data loaders."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ml.common.training import balanced_loader, ordered_loader


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

    def set_random_seed(self, seed: int) -> None:
        if hasattr(self.transform, "set_random_seed"):
            self.transform.set_random_seed(seed)


def train_loader(frame: pd.DataFrame, root: Path, transform, batch_size: int, workers: int, seed: int) -> DataLoader:
    frame = frame.reset_index(drop=True)
    return balanced_loader(ManifestImages(frame, root, transform), frame, batch_size, workers, seed)


def eval_loader(frame: pd.DataFrame, root: Path, transform, batch_size: int, workers: int) -> DataLoader:
    return ordered_loader(ManifestImages(frame, root, transform), batch_size, workers)
