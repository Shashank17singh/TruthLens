"""Manifest-backed audio dataset: a 4-second clip per item, randomly placed and augmented in training."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ml.audio.augment import Augmenter
from ml.audio.preprocess import fixed_clip, load_mono
from ml.common.training import balanced_loader, ordered_loader


class ManifestAudio(Dataset):
    def __init__(self, frame: pd.DataFrame, root: Path, sample_rate: int, clip_seconds: float, augmenter: Augmenter | None = None):
        self.frame = frame.reset_index(drop=True)
        self.root = root
        self.sample_rate = sample_rate
        self.length = int(clip_seconds * sample_rate)
        self.augmenter = augmenter

    def __len__(self) -> int:
        return len(self.frame)

    def load(self, index: int) -> np.ndarray:
        row = self.frame.iloc[index]
        return load_mono(self.root / row["source_dataset"] / row["path"], self.sample_rate)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, float]:
        wave = self.load(index)
        if self.augmenter is None:
            clip = fixed_clip(wave, self.length)  # from the start, so evaluation repeats exactly
        else:
            clip = self.augmenter(fixed_clip(self.augmenter.before_crop(wave), self.length, self.augmenter.rng))
        return torch.from_numpy(np.ascontiguousarray(clip, dtype=np.float32)), float(self.frame.at[index, "label"])

    def set_random_seed(self, seed: int) -> None:
        if self.augmenter is not None:
            self.augmenter.set_random_seed(seed)


def train_loader(frame: pd.DataFrame, root: Path, config: dict, batch_size: int) -> DataLoader:
    frame = frame.reset_index(drop=True)
    augmenter = Augmenter(config["augment"], config["sample_rate"], config["seed"])
    dataset = ManifestAudio(frame, root, config["sample_rate"], config["clip_seconds"], augmenter)
    return balanced_loader(dataset, frame, batch_size, config["num_workers"], config["seed"])


def eval_loader(frame: pd.DataFrame, root: Path, config: dict, batch_size: int) -> DataLoader:
    return ordered_loader(ManifestAudio(frame, root, config["sample_rate"], config["clip_seconds"]), batch_size, config["num_workers"])
