"""Image dataset adapters. Each one reads the raw files and yields manifest records."""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from ml.common.manifest import FAKE, REAL, REAL_GENERATOR, Record
from ml.common.splits import connect_groups

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
FACEFORENSICS_METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter")
GENIMAGE_REAL_FOLDER = "nature"
GENIMAGE_FAKE_FOLDER = "ai"


def faceforensics_records(raw: Path, entry: dict) -> Iterator[Record]:
    """Link each swap's two videos, so both identities stay in one split.

    Fake `000_003.mp4` joins source videos 000 and 003. Grouping on one id alone
    would let the other person appear in a different split.
    """
    quality = entry.get("quality", "c23")
    fakes = [
        (method, path)
        for method in FACEFORENSICS_METHODS
        for path in sorted((raw / "manipulated_sequences" / method / quality / "videos").glob("*.mp4"))
    ]
    groups = connect_groups(tuple(path.stem.split("_")[:2]) for _, path in fakes)

    for path in sorted((raw / "original_sequences" / "youtube" / quality / "videos").glob("*.mp4")):
        yield Record(
            path=str(path),
            label=REAL,
            source_dataset="faceforensics",
            generator_name=REAL_GENERATOR,
            group_id=groups.get(path.stem, path.stem),
            domain="face",
        )
    for method, path in fakes:
        yield Record(
            path=str(path),
            label=FAKE,
            source_dataset="faceforensics",
            generator_name=method,
            group_id=groups[path.stem.split("_")[0]],
            domain="face",
        )


def celebdf_records(raw: Path, entry: dict) -> Iterator[Record]:
    folders = (
        ("Celeb-real", REAL, REAL_GENERATOR),
        ("YouTube-real", REAL, REAL_GENERATOR),
        ("Celeb-synthesis", FAKE, "celeb_df_v2"),
    )
    for folder, label, generator in folders:
        for path in sorted((raw / folder).glob("*.mp4")):
            yield Record(
                path=str(path),
                label=label,
                source_dataset="celebdf",
                generator_name=generator,
                group_id=path.stem.split("_")[0],
                domain="face",
            )


def genimage_records(raw: Path, entry: dict) -> Iterator[Record]:
    """Generator folders hold `ai` and `nature` directories at varying depth."""
    for generator_dir in sorted(path for path in raw.iterdir() if path.is_dir()):
        generator = generator_dir.name.lower()
        for directory in sorted(path for path in generator_dir.rglob("*") if path.is_dir()):
            if directory.name not in (GENIMAGE_REAL_FOLDER, GENIMAGE_FAKE_FOLDER):
                continue
            is_real = directory.name == GENIMAGE_REAL_FOLDER
            for path in sorted(directory.iterdir()):
                if path.suffix.lower() in IMAGE_SUFFIXES:
                    yield Record(
                        path=str(path),
                        label=REAL if is_real else FAKE,
                        source_dataset="genimage",
                        generator_name=REAL_GENERATOR if is_real else generator,
                        group_id=path.stem,
                        domain="whole_image",
                    )


def aivr_records(raw: Path, entry: dict) -> Iterator[Record]:
    """`train` and `test` folders with `fake` and `real` inside. Fakes mix Stable Diffusion, Midjourney and DALL-E without labels."""
    for path in sorted(raw.glob("*/*/*")):
        split_dir, folder = path.parent.parent.name, path.parent.name
        if folder not in ("fake", "real") or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        is_real = folder == "real"
        yield Record(
            path=str(path),
            label=REAL if is_real else FAKE,
            source_dataset="aivr",
            generator_name=REAL_GENERATOR if is_real else "aivr_mixed",
            # File names restart at 0001 in every folder, so the folder is part of the group.
            group_id=f"aivr_{split_dir}_{folder}_{path.stem}",
            domain="whole_image",
        )


def shutterstock_records(raw: Path, entry: dict) -> Iterator[Record]:
    """Shutterstock photos paired with AI versions; `train.csv` maps each file to label 1 (AI) or 0 (real)."""
    labels = pd.read_csv(raw / "train.csv")
    for file_name, label in zip(labels["file_name"], labels["label"]):
        path = raw / file_name
        if not path.exists():
            continue
        is_real = int(label) == 0
        yield Record(
            path=str(path),
            label=REAL if is_real else FAKE,
            source_dataset="shutterstock",
            generator_name=REAL_GENERATOR if is_real else "shutterstock_ai",
            group_id=f"shutterstock_{path.stem}",
            domain="whole_image",
        )
