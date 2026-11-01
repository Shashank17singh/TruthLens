"""Audio preparation: 16 kHz mono, silence trimmed, cut to 4-second clips at load time."""

import hashlib
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import partial
from pathlib import Path

import librosa
import numpy as np
import soundfile
import soxr

from ml.common.checksums import hash_file
from ml.common.manifest import Record

SILENCE_PEAK = 1e-4


def load_mono(path: Path, sample_rate: int) -> np.ndarray:
    wave, source_rate = soundfile.read(path, dtype="float32", always_2d=True)
    mono = wave.mean(axis=1)
    if source_rate != sample_rate:
        mono = soxr.resample(mono, source_rate, sample_rate)
    return mono.astype(np.float32)


def trim_silence(wave: np.ndarray, top_db: float) -> np.ndarray:
    """Trim relative to the file's peak, which needs a separate check for silent files."""
    if wave.size == 0 or np.abs(wave).max() < SILENCE_PEAK:
        return wave[:0]
    trimmed, _ = librosa.effects.trim(wave, top_db=top_db)
    return trimmed


def fixed_clip(wave: np.ndarray, length: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Return exactly `length` samples: repeat short audio, crop long audio.

    Pass `rng` for a random crop during training. Without it the clip is taken
    from the start, so evaluation is repeatable.
    """
    if wave.size == 0:
        raise ValueError("cannot make a clip from empty audio")
    if wave.size < length:
        return np.tile(wave, length // wave.size + 1)[:length]
    start = 0 if rng is None else int(rng.integers(0, wave.size - length + 1))
    return wave[start : start + length]


def _prepare_one(record: Record, output_dir: Path, config: dict) -> tuple[Record | None, str]:
    sample_rate = config["sample_rate"]
    try:
        wave = trim_silence(load_mono(Path(record.path), sample_rate), config["trim_top_db"])
    except (soundfile.LibsndfileError, RuntimeError, ValueError):
        return None, "dropped_unreadable"
    if wave.size < config["min_seconds"] * sample_rate:
        return None, "dropped_too_short"

    name = hashlib.sha256(f"{record.source_dataset}/{record.path}".encode()).hexdigest()[:24]
    relative = f"audio/{name[:2]}/{name}.flac"
    target = output_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(target, wave, sample_rate, subtype="PCM_16", format="FLAC")
    return replace(record, path=relative, content_hash=hash_file(target)), "output_records"


def prepare(
    records: Sequence[Record], output_dir: Path, config: dict, workers: int = 1
) -> tuple[list[Record], dict[str, int]]:
    stats: Counter = Counter(input_records=len(records))
    work = partial(_prepare_one, output_dir=output_dir, config=config)
    if workers <= 1:
        results = map(work, records)
        prepared = list(results)
    else:
        with ProcessPoolExecutor(workers) as pool:
            prepared = list(pool.map(work, records, chunksize=64))

    kept = []
    for record, outcome in prepared:
        stats[outcome] += 1
        if record is not None:
            kept.append(record)
    return kept, dict(stats)
