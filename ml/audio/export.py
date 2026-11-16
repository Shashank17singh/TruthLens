"""Export an audio checkpoint, verify it against PyTorch, and time a 30-second clip on CPU.

    python -m ml.audio.export checkpoints/audio/<run>/<run>-best.pt

The graph takes raw 16 kHz audio [batch, 64000], because features are computed
inside the model. ONNX is tried first; a model that fails to export or to
match falls back to TorchScript, as the prompt allows for wav2vec2.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from ml.audio.data import ManifestAudio
from ml.audio.inference import score_long
from ml.audio.model import load_checkpoint
from ml.common.checksums import hash_file
from ml.common.manifest import read_manifest
from ml.image.export import TOLERANCE, TRACE_BATCH, ExportMismatch, session

VERIFY_BATCH_SIZES = (1, 4, 2)
VERIFY_CLIPS = sum(VERIFY_BATCH_SIZES)
BENCHMARK_SECONDS = 30
WARMUP_RUNS = 2
BENCHMARK_RUNS = 10


def generated_speech(count: int, samples: int, sample_rate: int, seed: int) -> np.ndarray:
    """Voiced-speech-like signals: a wandering pitch with harmonics, syllable-rate envelope and breath noise.

    Used when no real clips are available. Like images, raw white noise is far
    from the training data and can drive logits to extremes where rounding
    alone breaks the tolerance.
    """
    rng = np.random.default_rng(seed)
    time_axis = np.arange(samples) / sample_rate
    clips = []
    for _ in range(count):
        pitch = rng.uniform(90, 220) * (1 + 0.08 * np.sin(2 * np.pi * rng.uniform(0.5, 3) * time_axis))
        phase = 2 * np.pi * np.cumsum(pitch) / sample_rate
        voiced = sum(np.sin(harmonic * phase) / harmonic for harmonic in range(1, 12))
        envelope = 0.5 * (1 + np.sin(2 * np.pi * rng.uniform(3, 6) * time_axis + rng.uniform(0, np.pi)))
        clip = voiced * envelope + 0.02 * rng.standard_normal(samples)
        clips.append(0.3 * clip / np.abs(clip).max())
    return np.stack(clips).astype(np.float32)


def verification_clips(checkpoint: dict) -> tuple[np.ndarray, str]:
    config = checkpoint["config"]
    samples = int(config["clip_seconds"] * config["sample_rate"])
    manifest = Path(config["manifest"])
    if manifest.exists():
        frame = read_manifest(manifest)
        frame = frame[frame["split"] == "val"].head(VERIFY_CLIPS)
        if len(frame) == VERIFY_CLIPS:
            dataset = ManifestAudio(frame, Path(config["data_root"]), config["sample_rate"], config["clip_seconds"])
            return np.stack([dataset[index][0].numpy() for index in range(VERIFY_CLIPS)]), "validation clips"
    return generated_speech(VERIFY_CLIPS, samples, config["sample_rate"], config["seed"]), "generated speech-like clips"


class Runner:
    """One interface over ONNX Runtime and TorchScript, so verification and timing treat both alike."""

    def __init__(self, path: Path, format_: str, threads: int | None = None):
        self.format = format_
        if format_ == "onnx":
            self.session = session(path, threads)
        else:
            if threads:
                torch.set_num_threads(threads)
            self.module = torch.jit.load(str(path)).eval()

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        if self.format == "onnx":
            return self.session.run(["logit"], {"wave": batch})[0]
        with torch.inference_mode():
            return self.module(torch.from_numpy(batch)).numpy()


def export_onnx(model: torch.nn.Module, path: Path, samples: int) -> None:
    torch.onnx.export(
        model.eval(), (torch.zeros(TRACE_BATCH, samples),), path,
        input_names=["wave"], output_names=["logit"], dynamic_shapes=({0: "batch"},), external_data=False, dynamo=True,
    )


def export_torchscript(model: torch.nn.Module, path: Path, samples: int) -> None:
    traced = torch.jit.trace(model.eval(), torch.zeros(TRACE_BATCH, samples), check_trace=False)
    traced.save(str(path))


def verify(model: torch.nn.Module, runner: Runner, clips: np.ndarray) -> dict:
    largest, highest = 0.0, 0.0
    start = 0
    for size in VERIFY_BATCH_SIZES:
        batch = clips[start : start + size]
        start += size
        with torch.inference_mode():
            expected = model(torch.from_numpy(batch)).numpy()
        actual = runner(batch)
        if actual.shape != expected.shape:
            raise ExportMismatch(f"output shape {actual.shape} differs from {expected.shape}")
        largest = max(largest, float(np.abs(actual - expected).max()))
        highest = max(highest, float(np.abs(expected).max()))
    if largest > TOLERANCE:
        raise ExportMismatch(f"largest difference {largest:.2e} is above {TOLERANCE:.0e}")
    return {"max_output_difference": largest, "max_abs_logit": highest}


def benchmark(runner: Runner, sample_rate: int, clip_seconds: float, threads: int) -> dict:
    """Wall time to score a 30-second clip end to end: windowing, inference on 14 windows, aggregation."""
    wave = generated_speech(1, BENCHMARK_SECONDS * sample_rate, sample_rate, 0)[0]
    for _ in range(WARMUP_RUNS):
        score_long(runner, wave, sample_rate, clip_seconds)
    timings = []
    for _ in range(BENCHMARK_RUNS):
        started = time.perf_counter()
        score_long(runner, wave, sample_rate, clip_seconds)
        timings.append((time.perf_counter() - started) * 1000.0)
    return {"threads": threads, "clip_seconds": BENCHMARK_SECONDS, "runs": BENCHMARK_RUNS, "median_ms": round(statistics.median(timings), 1)}


def export_model(model: torch.nn.Module, name: str, config: dict, clips: np.ndarray, inputs: str, output_dir: Path, thread_counts: tuple[int, ...]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = int(config["clip_seconds"] * config["sample_rate"])
    attempts = []
    for format_, suffix, exporter in (("onnx", ".onnx", export_onnx), ("torchscript", ".ts.pt", export_torchscript)):
        path = output_dir / f"{name}{suffix}"
        try:
            exporter(model, path, samples)
            parity = verify(model, Runner(path, format_), clips)
            break
        except Exception as error:  # noqa: BLE001 - an exporter crash and a parity failure both mean: try the next format
            attempts.append({"format": format_, "error": f"{type(error).__name__}: {str(error)[:300]}"})
            path.unlink(missing_ok=True)
    else:
        raise ExportMismatch(f"no export format worked: {attempts}")
    report = {
        "name": name,
        "format": format_,
        "file": path.name,
        "sha256": hash_file(path),
        "size_mb": round(path.stat().st_size / 1_000_000, 1),
        "sample_rate": config["sample_rate"],
        "clip_seconds": config["clip_seconds"],
        "window_overlap": 0.5,
        "failed_attempts": attempts,
        **parity,
        "verification_inputs": inputs,
        "tolerance": TOLERANCE,
        "latency_30s": [benchmark(Runner(path, format_, threads), config["sample_rate"], config["clip_seconds"], threads) for threads in thread_counts],
    }
    (output_dir / f"{name}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def export_checkpoint(checkpoint_path: Path, output_dir: Path, thread_counts: tuple[int, ...] = (1, 4)) -> dict:
    model, checkpoint = load_checkpoint(checkpoint_path)
    clips, inputs = verification_clips(checkpoint)
    return export_model(model, checkpoint["name"], checkpoint["config"], clips, inputs, output_dir, thread_counts)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/onnx"))
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.checkpoint, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
