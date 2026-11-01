"""The manifest is the single contract between data preparation and training."""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

COLUMNS = (
    "path",
    "label",
    "source_dataset",
    "generator_name",
    "group_id",
    "domain",
    "content_hash",
    "split",
)
SPLITS = ("train", "val", "test", "unseen_test")

REAL = 0
FAKE = 1
REAL_GENERATOR = "real"
UNKNOWN_GENERATOR = "unknown"


class ManifestError(ValueError):
    """A manifest breaks the schema."""


class LeakageError(ValueError):
    """The same content, identity, speaker or prompt reached two splits."""


@dataclass(frozen=True, slots=True)
class Record:
    """One sample. `path` is the raw file until preprocessing replaces it.

    `split` is set only when a benchmark defines its own split. `text` is set
    only for text samples, which are stored in a Parquet file by preprocessing.
    """

    path: str
    label: int
    source_dataset: str
    generator_name: str
    group_id: str
    domain: str = ""
    content_hash: str = ""
    split: str = ""
    text: str | None = None


def records_to_frame(records: Iterable[Record]) -> pd.DataFrame:
    rows = [{column: getattr(record, column) for column in COLUMNS} for record in records]
    return pd.DataFrame(rows, columns=list(COLUMNS))


def validate(frame: pd.DataFrame) -> None:
    missing = set(COLUMNS) - set(frame.columns)
    if missing:
        raise ManifestError(f"missing columns: {sorted(missing)}")
    if not frame["label"].isin([REAL, FAKE]).all():
        raise ManifestError("labels must be 0 (real) or 1 (fake)")
    is_real_generator = frame["generator_name"] == REAL_GENERATOR
    if (is_real_generator != (frame["label"] == REAL)).any():
        raise ManifestError(f"generator_name must be {REAL_GENERATOR!r} exactly for real samples")
    if not frame["split"].isin(SPLITS).all():
        raise ManifestError(f"split must be one of {SPLITS}")
    for column in ("path", "source_dataset", "generator_name", "group_id", "content_hash"):
        if (frame[column] == "").any():
            raise ManifestError(f"{column} must not be empty")
    if frame["path"].duplicated().any():
        raise ManifestError("each path may appear only once")


def check_no_leakage(frame: pd.DataFrame) -> None:
    """Raise when anything is shared across splits.

    A group may span `test` and `unseen_test`, since neither is used for fitting.
    Identical content may not, because it would be counted twice.
    """
    sides = frame["split"].replace("unseen_test", "test")
    for column, splits, meaning in (
        ("content_hash", frame["split"], "identical content"),
        ("group_id", sides, "a group"),
    ):
        spread = splits.groupby(frame[column], sort=False).nunique()
        leaked = spread[spread > 1]
        if len(leaked):
            raise LeakageError(f"{len(leaked)} cases of {meaning} span splits, e.g. {leaked.index[0]}")

    unseen = set(frame.loc[frame["split"] == "unseen_test", "generator_name"])
    fitted = set(frame.loc[frame["split"].isin(["train", "val"]), "generator_name"])
    overlap = (unseen & fitted) - {REAL_GENERATOR}
    if overlap:
        raise LeakageError(f"unseen generators were used in training: {sorted(overlap)}")

    conflicting = frame.groupby("content_hash", sort=False)["label"].nunique()
    if (conflicting > 1).any():
        raise LeakageError("identical content carries both labels")


def write_manifest(frame: pd.DataFrame, path: Path) -> None:
    validate(frame)
    check_no_leakage(frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, list(COLUMNS)].sort_values("path").to_csv(path, index=False, lineterminator="\n")


def read_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={c: str for c in COLUMNS if c != "label"}, keep_default_na=False)
    validate(frame)
    return frame


def combine(paths: Sequence[Path]) -> pd.DataFrame:
    """Concatenate manifests and re-check leakage across datasets."""
    frames = [read_manifest(path) for path in paths]
    combined = pd.concat(frames, ignore_index=True)
    validate(combined)
    check_no_leakage(combined)
    return combined


def unseen_evaluation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Unseen fakes plus the real test samples, since AUC needs both classes."""
    is_unseen = frame["split"] == "unseen_test"
    is_real_test = (frame["split"] == "test") & (frame["label"] == REAL)
    return frame[is_unseen | is_real_test].reset_index(drop=True)
