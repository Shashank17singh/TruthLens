"""Class balance reporting and weighted sampling."""

import numpy as np
import pandas as pd


def balance_report(frame: pd.DataFrame) -> dict:
    """Counts by split, label and generator, for the data card and for review."""
    by_split = {}
    for split, rows in frame.groupby("split", sort=True):
        fake_share = float(rows["label"].mean())
        by_split[split] = {
            "samples": len(rows),
            "real": int((rows["label"] == 0).sum()),
            "fake": int((rows["label"] == 1).sum()),
            "fake_share": round(fake_share, 4),
            "groups": int(rows["group_id"].nunique()),
        }
    generators = (
        frame.groupby(["generator_name", "split"]).size().unstack(fill_value=0).sort_index()
    )
    return {
        "total": len(frame),
        "splits": by_split,
        "generators": {name: {k: int(v) for k, v in row.items()} for name, row in generators.iterrows()},
    }


def sample_weights(frame: pd.DataFrame) -> np.ndarray:
    """Weights giving each label half the draws, shared equally by its generators.

    Balancing only real against fake would still let a large generator crowd
    out a small one, and the model would mostly learn the large one's artefacts.
    """
    counts = frame.groupby(["label", "generator_name"])["path"].transform("size")
    generators_in_label = frame.groupby("label")["generator_name"].transform("nunique")
    labels_present = frame["label"].nunique()
    weights = 1.0 / (labels_present * generators_in_label * counts)
    return weights.to_numpy(dtype=np.float64, copy=True)


def weighted_sampler(frame: pd.DataFrame, seed: int):
    """A seeded PyTorch sampler that draws one epoch's worth of samples."""
    import torch
    from torch.utils.data import WeightedRandomSampler

    generator = torch.Generator().manual_seed(seed)
    weights = torch.as_tensor(sample_weights(frame), dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(frame), replacement=True, generator=generator)
