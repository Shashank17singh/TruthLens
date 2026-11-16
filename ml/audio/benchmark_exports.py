"""Export every audio architecture, verify it, and time a 30-second clip on CPU.

    python -m ml.audio.benchmark_exports --output ml/eval/results/audio_latency.csv

ResNet-18 uses ImageNet weights, wav2vec2 and WavLM their published
self-supervised weights, and LCNN random weights, since it has no pretrained
version. Latency depends on the architecture, not on what was learned, so
these numbers are valid before training. They say nothing about accuracy.
"""

import argparse
import platform
from pathlib import Path

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from ml.audio.export import VERIFY_CLIPS, export_model, generated_speech
from ml.audio.model import build_model

CONFIG_DIR = str(Path(__file__).parent / "configs")
VARIANTS = [
    *((model, feature) for model in ("resnet18", "lcnn") for feature in ("mel", "lfcc", "cqt")),
    ("wav2vec2", None),
    ("wavlm", None),
]


def variant_config(model: str, feature: str | None) -> dict:
    overrides = [f"model={model}"] + ([f"feature={feature}"] if feature else [])
    with initialize_config_dir(CONFIG_DIR, version_base=None):
        return OmegaConf.to_container(compose("train", overrides=overrides), resolve=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("ml/eval/results/audio_latency.csv"))
    parser.add_argument("--export-dir", type=Path, default=Path("checkpoints/onnx_benchmark"))
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--only", nargs="+", help="model names to run, e.g. lcnn wav2vec2")
    args = parser.parse_args()

    rows = []
    for model_name, feature in VARIANTS:
        if args.only and model_name not in args.only:
            continue
        config = variant_config(model_name, feature)
        torch.manual_seed(1337)
        model = build_model(config).eval()
        samples = int(config["clip_seconds"] * config["sample_rate"])
        clips = generated_speech(VERIFY_CLIPS, samples, config["sample_rate"], seed=1337)
        name = f"bench-audio-{model_name}-{feature or 'wave'}"
        report = export_model(model, name, config, clips, "generated speech-like clips", args.export_dir, tuple(args.threads))
        row = {
            "model": model_name,
            "input": feature or "waveform",
            "weights": "random" if model_name == "lcnn" else "pretrained",
            "format": report["format"],
            "size_mb": report["size_mb"],
            "max_output_difference": report["max_output_difference"],
            "machine": platform.processor() or platform.machine(),
        }
        for latency in report["latency_30s"]:
            row[f"median_ms_30s_{latency['threads']}t"] = latency["median_ms"]
        rows.append(row)
        print(row, flush=True)
        save(rows, args.output)  # after every variant, so a crash on a large model keeps the rest
        del model
    print(f"latency table: {args.output}")


def save(rows: list[dict], output: Path) -> None:
    """Merge into an existing table: a rerun of some variants replaces only their rows."""
    table = pd.DataFrame(rows)
    if output.exists():
        previous = pd.read_csv(output)
        keep = ~previous.set_index(["model", "input"]).index.isin(table.set_index(["model", "input"]).index)
        table = pd.concat([previous[keep], table], ignore_index=True)
    order = {(model, feature or "waveform"): index for index, (model, feature) in enumerate(VARIANTS)}
    rank = [order.get(pair, len(order)) for pair in zip(table["model"], table["input"])]
    table = table.assign(_rank=rank).sort_values("_rank", kind="stable").drop(columns="_rank")
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False, lineterminator="\n")


if __name__ == "__main__":
    main()
