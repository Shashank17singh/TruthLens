"""Seed control shared by every pipeline and training entry point."""

import os
import random

import numpy as np

DEFAULT_SEED = 1337


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Seed Python, NumPy and PyTorch when PyTorch is installed."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
