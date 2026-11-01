"""Audio dataset adapters. Each one reads the raw files and yields manifest records."""

import csv
from collections.abc import Iterator
from pathlib import Path

from ml.common.manifest import FAKE, REAL, REAL_GENERATOR, UNKNOWN_GENERATOR, Record

ASVSPOOF_SPLITS = (
    ("train", "train", "ASVspoof2019.LA.cm.train.trn.txt"),
    ("dev", "val", "ASVspoof2019.LA.cm.dev.trl.txt"),
    ("eval", "test", "ASVspoof2019.LA.cm.eval.trl.txt"),
)


def asvspoof2019_la_records(raw: Path, entry: dict) -> Iterator[Record]:
    root = raw / "LA"
    for folder, split, protocol in ASVSPOOF_SPLITS:
        lines = (root / "ASVspoof2019_LA_cm_protocols" / protocol).read_text().splitlines()
        for line in lines:
            speaker, utterance, _, attack, key = line.split()
            is_real = key == "bonafide"
            yield Record(
                path=str(root / f"ASVspoof2019_LA_{folder}" / "flac" / f"{utterance}.flac"),
                label=REAL if is_real else FAKE,
                source_dataset="asvspoof2019_la",
                generator_name=REAL_GENERATOR if is_real else attack,
                group_id=speaker,
                split=split,
            )


def asvspoof2021_df_records(raw: Path, entry: dict) -> Iterator[Record]:
    """Keep the `eval` phase, which is the published benchmark subset."""
    metadata = raw / "keys" / "DF" / "CM" / "trial_metadata.txt"
    for line in metadata.read_text().splitlines():
        fields = line.split()
        speaker, utterance, codec, _, attack, key, _, phase = fields[:8]
        if phase != "eval":
            continue
        is_real = key == "bonafide"
        yield Record(
            path=str(raw / "ASVspoof2021_DF_eval" / "flac" / f"{utterance}.flac"),
            label=REAL if is_real else FAKE,
            source_dataset="asvspoof2021_df",
            generator_name=REAL_GENERATOR if is_real else attack,
            group_id=speaker,
            domain=codec,
        )


def wavefake_records(raw: Path, entry: dict) -> Iterator[Record]:
    """Pair LJSpeech with its vocoded copies. JSUT is skipped: its real audio is not bundled."""
    for path in sorted((raw / "LJSpeech-1.1" / "wavs").glob("*.wav")):
        yield Record(
            path=str(path),
            label=REAL,
            source_dataset="wavefake",
            generator_name=REAL_GENERATOR,
            group_id=path.stem,
        )
    for folder in sorted((raw / "generated_audio").glob("ljspeech_*")):
        for path in sorted(folder.glob("*.wav")):
            yield Record(
                path=str(path),
                label=FAKE,
                source_dataset="wavefake",
                generator_name=folder.name.removeprefix("ljspeech_"),
                group_id=path.stem.removesuffix("_gen"),
            )


def in_the_wild_records(raw: Path, entry: dict) -> Iterator[Record]:
    with (raw / "meta.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            is_real = row["label"] == "bona-fide"
            yield Record(
                path=str(raw / row["file"]),
                label=REAL if is_real else FAKE,
                source_dataset="in_the_wild",
                generator_name=REAL_GENERATOR if is_real else UNKNOWN_GENERATOR,
                group_id=row["speaker"],
            )
