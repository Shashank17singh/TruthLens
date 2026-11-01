"""Export a checkpoint to ONNX, verify it against PyTorch, and measure CPU latency.

    python -m ml.image.export checkpoints/image/<run>/<run>-best.pt
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F

from ml.common.checksums import hash_file
from ml.common.manifest import read_manifest
from ml.image.augment import eval_transform
from ml.image.data import ManifestImages
from ml.image.model import ImageDetector, load_checkpoint

TOLERANCE = 1e-3
# Tracing with one image hard-codes the batch size into the graph, even when it is
# declared dynamic, so the trace must use a larger batch.
TRACE_BATCH = 2
VERIFY_BATCH_SIZES = (1, 4, 2)
VERIFY_SAMPLES = sum(VERIFY_BATCH_SIZES)
GENERATED_GRID = 16
WARMUP_RUNS = 5
BENCHMARK_RUNS = 50


class ExportMismatch(ValueError):
    """ONNX and PyTorch outputs differ by more than the tolerance."""


def export_onnx(model: ImageDetector, path: Path, image_size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(TRACE_BATCH, 3, image_size, image_size)
    torch.onnx.export(
        model.eval(),
        (dummy,),
        path,
        input_names=["image"],
        output_names=["logit"],
        dynamic_shapes=({0: "batch"},),
        external_data=False,
        dynamo=True,
    )


def session(path: Path, threads: int | None = None) -> onnxruntime.InferenceSession:
    options = onnxruntime.SessionOptions()
    if threads:
        options.intra_op_num_threads = threads
    return onnxruntime.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def generated_images(count: int, size: int, mean: tuple[float, ...], std: tuple[float, ...], seed: int) -> torch.Tensor:
    """Smooth images in the valid pixel range, used when no real images are available."""
    generator = torch.Generator().manual_seed(seed)
    coarse = torch.rand(count, 3, GENERATED_GRID, GENERATED_GRID, generator=generator)
    pixels = F.interpolate(coarse, size=(size, size), mode="bicubic", align_corners=False).clamp(0.0, 1.0)
    return (pixels - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)


def verification_images(checkpoint: dict) -> tuple[torch.Tensor, str]:
    """Verify on images, not on raw noise.

    Noise is far outside what the model was trained on. A small model can return
    logits near -1000 on it, where ordinary float32 rounding alone exceeds any
    useful tolerance, so noise would fail a correct export.
    """
    config = checkpoint["config"]
    manifest = Path(config["manifest"])
    if manifest.exists():
        frame = read_manifest(manifest)
        frame = frame[frame["split"] == "val"].head(VERIFY_SAMPLES)
        dataset = ManifestImages(frame, Path(config["data_root"]), eval_transform(checkpoint["mean"], checkpoint["std"]))
        if len(dataset) == VERIFY_SAMPLES:
            return torch.stack([dataset[index][0] for index in range(VERIFY_SAMPLES)]), "validation images"
    images = generated_images(VERIFY_SAMPLES, config["image_size"], checkpoint["mean"], checkpoint["std"], config["seed"])
    return images, "generated images"


def verify(model: ImageDetector, path: Path, images: torch.Tensor) -> dict:
    """Compare both runtimes across batch sizes. Returns the largest difference and logit range."""
    runtime = session(path)
    largest, highest = 0.0, 0.0
    for batch in torch.split(images, list(VERIFY_BATCH_SIZES)):
        with torch.inference_mode():
            expected = model(batch).numpy()
        actual = runtime.run(["logit"], {"image": batch.numpy()})[0]
        if actual.shape != expected.shape:
            raise ExportMismatch(f"output shape {actual.shape} differs from {expected.shape}")
        largest = max(largest, float(np.abs(actual - expected).max()))
        highest = max(highest, float(np.abs(expected).max()))
    if largest > TOLERANCE:
        raise ExportMismatch(f"largest difference {largest:.2e} is above {TOLERANCE:.0e}")
    return {"max_output_difference": largest, "max_abs_logit": highest}


def benchmark(path: Path, image_size: int, threads: int) -> dict:
    runtime = session(path, threads)
    image = np.random.default_rng(0).standard_normal((1, 3, image_size, image_size), dtype=np.float32)
    for _ in range(WARMUP_RUNS):
        runtime.run(["logit"], {"image": image})
    timings = []
    for _ in range(BENCHMARK_RUNS):
        started = time.perf_counter()
        runtime.run(["logit"], {"image": image})
        timings.append((time.perf_counter() - started) * 1000.0)
    timings.sort()
    return {
        "threads": threads,
        "runs": BENCHMARK_RUNS,
        "median_ms": round(statistics.median(timings), 2),
        "p95_ms": round(timings[int(0.95 * len(timings)) - 1], 2),
    }


def export_model(
    model: ImageDetector, name: str, image_size: int, images: torch.Tensor, inputs: str,
    output_dir: Path, thread_counts: tuple[int, ...],
) -> dict:
    path = output_dir / f"{name}.onnx"
    export_onnx(model, path, image_size)
    report = {
        "name": name,
        "onnx": path.name,
        "sha256": hash_file(path),
        "size_mb": round(path.stat().st_size / 1_000_000, 1),
        "image_size": image_size,
        "mean": list(model.mean),
        "std": list(model.std),
        **verify(model, path, images),
        "verification_inputs": inputs,
        "tolerance": TOLERANCE,
        "latency": [benchmark(path, image_size, threads) for threads in thread_counts],
    }
    path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def export_checkpoint(checkpoint_path: Path, output_dir: Path, thread_counts: tuple[int, ...] = (1, 4)) -> dict:
    model, checkpoint = load_checkpoint(checkpoint_path)
    images, inputs = verification_images(checkpoint)
    return export_model(
        model, checkpoint["name"], checkpoint["config"]["image_size"], images, inputs, output_dir, thread_counts
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/onnx"))
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.checkpoint, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
