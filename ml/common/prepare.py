"""Prepare a dataset end to end.

    python -m ml.common.prepare mage --workers 4
    python -m ml.common.prepare mage --limit 5000
    python -m ml.common.prepare --combine image_core faceforensics genimage
"""

import argparse
import dataclasses
import importlib
import json
import os
import random
import sys
from collections import defaultdict
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import yaml

from ml.common import balance, datacard
from ml.common.checksums import verify_or_record
from ml.common.download import download, extract
from ml.common.manifest import Record, combine, records_to_frame, write_manifest
from ml.common.seed import set_seed
from ml.common.splits import SplitConfig, assign_splits

COMMON_DIR = Path(__file__).parent
REPO_ROOT = COMMON_DIR.parents[1]
REGISTRY_PATH = COMMON_DIR / "datasets.yaml"
CONFIG_PATH = COMMON_DIR / "configs" / "preprocess.yaml"
ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tgz", ".tar.bz2")


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def fetch(name: str, entry: dict, raw: Path) -> None:
    """Download or locate raw files, verify their bytes and unpack archives."""
    for item in entry.get("files", []):
        target = download(item["url"], raw / item["name"], item.get("checksum"))
        if "checksum" not in item and verify_or_record(target, f"{name}/{item['name']}"):
            print(f"recorded checksum for {item['name']}")

    if entry["access"] == "manual":
        if not raw.is_dir() or not any(raw.iterdir()):
            sys.exit(f"{name} needs a manual download.\n{entry['instructions']}")
        for archive in sorted(p for p in raw.iterdir() if p.name.endswith(ARCHIVE_SUFFIXES)):
            if verify_or_record(archive, f"{name}/{archive.name}"):
                print(f"recorded checksum for {archive.name}")

    for archive in sorted(p for p in raw.iterdir() if p.name.endswith(ARCHIVE_SUFFIXES)):
        extract(archive, raw)


def stratified_limit(records: Sequence[Record], limit: int, seed: int) -> list[Record]:
    """Sample whole groups evenly across generators, so a subset stays leak-free."""
    by_stratum: dict[tuple[int, str], dict[str, list[Record]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_stratum[(record.label, record.generator_name)][record.group_id].append(record)

    rng = random.Random(seed)
    per_stratum = max(1, limit // len(by_stratum))
    chosen_groups: set[str] = set()
    for stratum in sorted(by_stratum):
        groups = sorted(by_stratum[stratum])
        rng.shuffle(groups)
        taken = 0
        for group in groups:
            if taken >= per_stratum:
                break
            chosen_groups.add(group)
            taken += len(by_stratum[stratum][group])
    return [record for record in records if record.group_id in chosen_groups]


def preprocess(modality: str, records: list[Record], output: Path, config: dict, workers: int):
    module = importlib.import_module(f"ml.{modality}.preprocess")
    return module.prepare(records, output, config[modality], workers)


def run(name: str, data_root: Path, cards_dir: Path, workers: int, limit: int | None) -> Path:
    registry = load_yaml(REGISTRY_PATH)
    if name not in registry:
        sys.exit(f"unknown dataset {name!r}. Choose from: {', '.join(sorted(registry))}")
    entry = registry[name]
    config = load_yaml(CONFIG_PATH)
    seed = config["seed"]
    set_seed(seed)

    raw = data_root / "raw" / name
    fetch(name, entry, raw)

    module_name, function_name = entry["adapter"].split(":")
    records = list(getattr(importlib.import_module(module_name), function_name)(raw, entry))
    if limit:
        records = stratified_limit(records, limit, seed)
    print(f"{name}: {len(records):,} records found")

    prepared, stats = preprocess(entry["modality"], records, data_root / "processed" / name, config, workers)
    frame = records_to_frame(prepared)

    split = entry["split"]
    result = assign_splits(
        frame,
        SplitConfig(
            strategy=split["strategy"],
            ratios=tuple(config["split_ratios"]),
            seed=seed,
            heldout_generators=frozenset(split.get("heldout_generators", [])),
        ),
    )
    stats = {**stats, "dropped_heldout_outside_test": result.dropped_heldout}

    manifest_path = data_root / "manifests" / f"{name}.csv"
    write_manifest(result.frame, manifest_path)
    report = balance.balance_report(result.frame)
    manifest_path.with_suffix(".balance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if limit is None:
        card = datacard.render(name, entry, report, stats, date.today())
        datacard.write(cards_dir / f"{name}.md", card)

    print(json.dumps({"stats": stats, "splits": report["splits"]}, indent=2))
    print(f"manifest: {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", nargs="?", help="dataset id from ml/common/datasets.yaml")
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--cards-dir", type=Path, default=REPO_ROOT / "docs" / "datacards")
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int, help="prepare a stratified subset; no data card is written")
    parser.add_argument("--combine", nargs="+", metavar=("OUTPUT", "DATASET"), help="merge prepared manifests")
    args = parser.parse_args()

    if args.combine:
        output, *names = args.combine
        manifests = args.data_root / "manifests"
        frame = combine([manifests / f"{name}.csv" for name in names])
        write_manifest(frame, manifests / f"{output}.csv")
        print(f"{output}: {len(frame):,} samples from {', '.join(names)}, no leakage found")
        return
    if not args.dataset:
        parser.error("give a dataset id or --combine")
    run(args.dataset, args.data_root, args.cards_dir, args.workers, args.limit)


if __name__ == "__main__":
    main()
