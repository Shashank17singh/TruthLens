"""Text cleaning: normalization, deduplication, length and language filters."""

import hashlib
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, LangDetectException, detect

from ml.common.manifest import Record

LANGUAGE_SAMPLE_CHARS = 1500
SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2, "": 3}

_NOT_ALPHANUMERIC = re.compile(r"[\W_]+")
_ZERO_WIDTH = re.compile("[​-‍⁠﻿]")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?%)\]}])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[{])\s+")
_SPACED_QUOTES = re.compile(r'"\s+([^"]*?)\s+"')
_SPACED_CONTRACTION = re.compile(r"\s+(n't|'s|'re|'ve|'ll|'d|'m)\b", re.IGNORECASE)


def normalize(text: str) -> str:
    """Remove formatting that identifies a source instead of an author.

    Tokenized corpora such as HC3 put spaces around punctuation on the human
    side only, so a detector could otherwise score well by reading the spacing.
    """
    text = _ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", text))
    text = " ".join(text.split())
    text = _SPACED_QUOTES.sub(r'"\1"', text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    return _SPACED_CONTRACTION.sub(r"\1", text)


def content_hash(text: str) -> str:
    """Hash letters and digits only, so case and punctuation edits still count as duplicates."""
    key = _NOT_ALPHANUMERIC.sub("", text.casefold())
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def word_count(text: str) -> int:
    return len(text.split())


def _init_language_worker() -> None:
    DetectorFactory.seed = 0


def _is_english(text: str) -> bool:
    try:
        return detect(text[:LANGUAGE_SAMPLE_CHARS]) == "en"
    except LangDetectException:
        return False


def english_flags(texts: Sequence[str], workers: int) -> list[bool]:
    if workers <= 1:
        _init_language_worker()
        return [_is_english(text) for text in texts]
    with ProcessPoolExecutor(workers, initializer=_init_language_worker) as pool:
        return list(pool.map(_is_english, texts, chunksize=256))


def prepare(
    records: Sequence[Record], output_dir: Path, config: dict, workers: int = 1
) -> tuple[list[Record], dict[str, int]]:
    """Clean text, write `texts.parquet`, and return records keyed by content hash."""
    stats: Counter = Counter(input_records=len(records))
    cleaned: list[Record] = []
    for record in records:
        text = normalize(record.text or "")
        if not config["min_words"] <= word_count(text) <= config["max_words"]:
            stats["dropped_length"] += 1
            continue
        cleaned.append(replace(record, text=text, content_hash=content_hash(text)))

    labels_by_hash: dict[str, set[int]] = {}
    for record in cleaned:
        labels_by_hash.setdefault(record.content_hash, set()).add(record.label)

    # An official test sample keeps its place, so duplicates leave train instead.
    unique: dict[str, Record] = {}
    for record in sorted(cleaned, key=lambda item: (SPLIT_PRIORITY[item.split], item.path)):
        if len(labels_by_hash[record.content_hash]) > 1:
            stats["dropped_conflicting_labels"] += 1
        elif record.content_hash in unique:
            stats["dropped_duplicates"] += 1
        else:
            unique[record.content_hash] = record

    candidates = list(unique.values())
    flags = english_flags([record.text or "" for record in candidates], workers)
    kept = [record for record, english in zip(candidates, flags) if english]
    stats["dropped_language"] = len(candidates) - len(kept)
    stats["output_records"] = len(kept)

    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {"path": [f"text/{r.content_hash}" for r in kept], "text": [r.text for r in kept]}
    )
    frame.to_parquet(output_dir / "texts.parquet", index=False)
    return [replace(r, path=f"text/{r.content_hash}", text=None) for r in kept], dict(stats)
