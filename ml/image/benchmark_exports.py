"""Export each architecture with ImageNet weights to verify export and measure CPU latency.

    python -m ml.image.benchmark_exports --output ml/eval/results/image_latency.csv

Latency depends on the architecture and not on what the model learned, so this
gives valid numbers before any training. It says nothing about accuracy.
"""

import argparse
import platform
from pathlib import Path

import pandas as pd
import torch

from ml.image.export import VERIFY_SAMPLES, export_model, generated_images
from ml.image.model import ImageDetector

BACKBONES = {
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "convnext_tiny": "convnext_tiny.fb_in1k",
    "vit_b16": "vit_base_patch16_224.augreg_in21k_ft_in1k",
}
IMAGE_SIZE = 224


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("ml/eval/results/image_latency.csv"))
    parser.add_argument("--onnx-dir", type=Path, default=Path("checkpoints/onnx_benchmark"))
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    args = parser.parse_args()

    rows = []
    for short_name, backbone in BACKBONES.items():
        for use_frequency in (False, True):
            torch.manual_seed(1337)
            model = ImageDetector(backbone, use_frequency, pretrained=True, image_size=IMAGE_SIZE).eval()
            images = generated_images(VERIFY_SAMPLES, IMAGE_SIZE, model.mean, model.std, seed=1337)
            branch = "freq" if use_frequency else "rgb"
            report = export_model(
                model, f"bench-{short_name}-{branch}", IMAGE_SIZE, images, "generated images",
                args.onnx_dir, tuple(args.threads),
            )
            row = {
                "model": short_name,
                "branches": "RGB + frequency" if use_frequency else "RGB",
                "size_mb": report["size_mb"],
                "max_output_difference": report["max_output_difference"],
                "machine": platform.processor() or platform.machine(),
            }
            for latency in report["latency"]:
                row[f"median_ms_{latency['threads']}t"] = latency["median_ms"]
                row[f"p95_ms_{latency['threads']}t"] = latency["p95_ms"]
            rows.append(row)
            print(row, flush=True)
            del model

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, lineterminator="\n")
    print(f"latency table: {args.output}")


if __name__ == "__main__":
    main()
