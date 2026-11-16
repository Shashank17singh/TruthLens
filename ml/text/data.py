"""Manifest-backed text dataset: one 512-token window per text and step.

Training takes a random window of each text, so long texts are seen in full
over several epochs. Validation takes the first window, which keeps early
stopping cheap; final evaluation scores every window (see `model.score_text`).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from ml.common.training import balanced_loader, ordered_loader
from ml.text.model import encode_windows


def read_texts(frame: pd.DataFrame, root: Path) -> pd.Series:
    """Text of every manifest row, from each dataset's texts.parquet."""
    wanted = frame["path"].tolist()
    # Read only the wanted rows: MAGE alone holds 330k texts.
    tables = [pd.read_parquet(root / name / "texts.parquet", filters=[("path", "in", wanted)]) for name in frame["source_dataset"].unique()]
    lookup = pd.concat(tables).drop_duplicates("path").set_index("path")["text"]
    return lookup.reindex(frame["path"]).reset_index(drop=True)


class ManifestText(Dataset):
    def __init__(self, frame: pd.DataFrame, root: Path, tokenizer: PreTrainedTokenizerBase, config: dict, random_window: bool):
        self.frame = frame.reset_index(drop=True)
        self.texts = read_texts(self.frame, root)
        self.tokenizer = tokenizer
        self.max_length = config["max_length"]
        self.stride = config["stride"]
        self.random_window = random_window
        self.rng = np.random.default_rng(config["seed"])

    def set_random_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, float]:
        windows = encode_windows(self.tokenizer, self.texts[index], self.max_length, self.stride)
        chosen = int(self.rng.integers(len(windows))) if self.random_window else 0
        return torch.from_numpy(windows[chosen]), float(self.frame.at[index, "label"])


def train_loader(frame: pd.DataFrame, root: Path, tokenizer, config: dict, batch_size: int) -> DataLoader:
    dataset = ManifestText(frame, root, tokenizer, config, random_window=True)
    return balanced_loader(dataset, dataset.frame, batch_size, config["num_workers"], config["seed"])


def eval_loader(frame: pd.DataFrame, root: Path, tokenizer, config: dict, batch_size: int) -> DataLoader:
    return ordered_loader(ManifestText(frame, root, tokenizer, config, random_window=False), batch_size, config["num_workers"])
