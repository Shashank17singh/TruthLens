"""Split assignment that keeps groups whole and generators balanced.

Group separation always wins over stratification, because it is what prevents
leakage. See D15 in docs/decisions.md.
"""

import random
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

import pandas as pd

from ml.common.seed import DEFAULT_SEED

TRAIN_SPLITS = ("train", "val", "test")
UNSEEN = "unseen_test"


@dataclass(frozen=True)
class SplitConfig:
    strategy: str = "grouped"
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    seed: int = DEFAULT_SEED
    heldout_generators: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class SplitResult:
    frame: pd.DataFrame
    dropped_heldout: int


def connect_groups(pairs: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Union-find over linked identities, such as a face swap's source and target."""
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for left, right in pairs:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[max(root_left, root_right)] = min(root_left, root_right)
    return {item: find(item) for item in parent}


def _allocate(compositions: dict[str, Counter], ratios: tuple[float, ...], seed: int) -> dict[str, str]:
    """Place each whole group in the split that still most needs its strata.

    Need is relative to each split's target, so the largest groups spread
    across splits instead of all landing in train.
    """
    totals: Counter = Counter()
    for composition in compositions.values():
        totals.update(composition)

    order = sorted(compositions)
    random.Random(seed).shuffle(order)
    order.sort(key=lambda group: -sum(compositions[group].values()))

    placed = [Counter() for _ in ratios]
    assignment: dict[str, str] = {}
    for group in order:
        composition = compositions[group]
        best_index, best_score = 0, float("-inf")
        for index, ratio in enumerate(ratios):
            score = sum(
                count / totals[stratum] * (1 - placed[index][stratum] / (ratio * totals[stratum]))
                for stratum, count in composition.items()
            )
            if score > best_score:
                best_index, best_score = index, score
        placed[best_index].update(composition)
        assignment[group] = TRAIN_SPLITS[best_index]
    return assignment


def _grouped(frame: pd.DataFrame, config: SplitConfig) -> SplitResult:
    is_heldout = frame["generator_name"].isin(config.heldout_generators)
    strata = frame["label"].astype(str) + "|" + frame["generator_name"]

    compositions: dict[str, Counter] = {}
    for group, stratum in zip(frame.loc[~is_heldout, "group_id"], strata[~is_heldout]):
        compositions.setdefault(group, Counter())[stratum] += 1
    if len(compositions) < len(TRAIN_SPLITS):
        raise ValueError(f"need at least {len(TRAIN_SPLITS)} groups, found {len(compositions)}")

    assignment = _allocate(compositions, config.ratios, config.seed)
    split = frame["group_id"].map(assignment)

    # A held-out row may stay only if its identity never reaches training.
    only_heldout = split.isna()
    in_test_group = split == "test"
    split = split.where(~is_heldout)
    split[is_heldout & (only_heldout | in_test_group)] = UNSEEN

    keep = split.notna()
    return SplitResult(frame[keep].assign(split=split[keep]), int((~keep).sum()))


def _official(frame: pd.DataFrame, config: SplitConfig) -> SplitResult:
    if not frame["split"].isin(TRAIN_SPLITS).all():
        raise ValueError("the official strategy needs a split on every record")
    is_heldout = frame["generator_name"].isin(config.heldout_generators)
    in_test = frame["split"] == "test"
    keep = ~is_heldout | in_test
    result = frame.copy()
    result.loc[is_heldout & in_test, "split"] = UNSEEN
    return SplitResult(result[keep], int((~keep).sum()))


def assign_splits(frame: pd.DataFrame, config: SplitConfig) -> SplitResult:
    if config.strategy == "grouped":
        return _grouped(frame, config)
    if config.strategy == "official":
        return _official(frame, config)
    if config.strategy == "evaluation":
        return SplitResult(frame.assign(split="test"), 0)
    raise ValueError(f"unknown split strategy {config.strategy!r}")
