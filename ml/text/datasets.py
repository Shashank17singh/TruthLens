"""Text dataset adapters. Each one reads the raw files and yields manifest records."""

import csv
import hashlib
import json
import re
import sys
from collections.abc import Iterator
from pathlib import Path

from ml.common.manifest import FAKE, REAL, REAL_GENERATOR, Record

csv.field_size_limit(sys.maxsize if sys.maxsize < 2**31 else 2**31 - 1)

MAGE_SPLITS = {"train.csv": "train", "valid.csv": "val", "test.csv": "test"}
_MAGE_MACHINE = re.compile(r"^(?P<domain>.+?)_machine_(?P<method>[a-z]+)_(?P<model>.+)$")
PROMPT_WORDS = 12
NATIVE_ESSAYS_FILE = "HewlettStudentEssay_real_88.json"


def prompt_group(text: str) -> str:
    """Group by opening words, since MAGE continuations start from the human text."""
    words = text.casefold().split()[:PROMPT_WORDS]
    return hashlib.sha256(" ".join(words).encode("utf-8")).hexdigest()[:16]


def parse_mage_source(source: str) -> tuple[str, str]:
    """Return (domain, generator) from a MAGE `src` value such as `wp_machine_topical_7B`."""
    if source.endswith("_human"):
        return source.removesuffix("_human"), REAL_GENERATOR
    match = _MAGE_MACHINE.match(source)
    if match is None:
        raise ValueError(f"unrecognised MAGE source {source!r}")
    return match["domain"], match["model"]


def mage_records(raw: Path, entry: dict) -> Iterator[Record]:
    for file_name, split in MAGE_SPLITS.items():
        with (raw / file_name).open(encoding="utf-8-sig", newline="") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                domain, generator = parse_mage_source(row["src"])
                # MAGE inverts the usual convention: its label 1 means human.
                label = REAL if row["label"] == "1" else FAKE
                if (label == REAL) != (generator == REAL_GENERATOR):
                    raise ValueError(f"{file_name} row {index}: label contradicts {row['src']!r}")
                yield Record(
                    path=f"{split}/{index}",
                    label=label,
                    source_dataset="mage",
                    generator_name=generator,
                    group_id=prompt_group(row["text"]),
                    domain=domain,
                    text=row["text"],
                )


def hc3_records(raw: Path, entry: dict) -> Iterator[Record]:
    with (raw / "all.jsonl").open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream):
            row = json.loads(line)
            group = hashlib.sha256(row["question"].encode("utf-8")).hexdigest()[:16]
            answers = (
                (row["human_answers"], REAL, REAL_GENERATOR),
                (row["chatgpt_answers"], FAKE, "chatgpt"),
            )
            for items, label, generator in answers:
                for index, text in enumerate(items):
                    yield Record(
                        path=f"{line_number}/{generator}/{index}",
                        label=label,
                        source_dataset="hc3",
                        generator_name=generator,
                        group_id=group,
                        domain=row["source"],
                        text=text,
                    )


def detector_bias_records(raw: Path, entry: dict) -> Iterator[Record]:
    rows = json.loads((raw / "all_data.json").read_text(encoding="utf-8"))
    for index, row in enumerate(rows):
        is_human = row["kind"] == "Human-Written"
        yield Record(
            path=f"{row['domain']}/{index}",
            label=REAL if is_human else FAKE,
            source_dataset="detector_bias",
            generator_name=REAL_GENERATOR if is_human else row["model"].lower(),
            group_id=f"{row['domain']}/{index}",
            domain=row["domain"],
            text=row["text"],
        )

    # The paper's native-writer baseline is published only in the authors' repository.
    essays = json.loads((raw / NATIVE_ESSAYS_FILE).read_text(encoding="utf-8"))
    domain = NATIVE_ESSAYS_FILE.removesuffix(".json")
    for index, row in enumerate(essays):
        yield Record(
            path=f"{domain}/{index}",
            label=REAL,
            source_dataset="detector_bias",
            generator_name=REAL_GENERATOR,
            group_id=f"{domain}/{index}",
            domain=domain,
            text=row["document"],
        )
