"""Long audio: score 4-second windows with 50% overlap, then aggregate by mean and by max.

Mean suits a clip that is fake throughout. Max catches a short spliced
segment inside real speech, at the cost of more false alarms on long real
recordings. Both are reported, and the timeline lets the UI show where.
"""

from collections.abc import Callable

import numpy as np

from ml.audio.preprocess import fixed_clip

OVERLAP = 0.5
# Four windows at a time: wav2vec2 on all 14 windows of a 30-second clip needs over 1 GB and
# failed with an allocation error on an 8 GB laptop. CNN models lose almost nothing.
BATCH = 4


def window_starts(samples: int, length: int, overlap: float = OVERLAP) -> list[int]:
    """Window start positions; the last window is moved back so the tail is always covered."""
    if samples <= length:
        return [0]
    hop = max(1, int(length * (1 - overlap)))
    starts = list(range(0, samples - length + 1, hop))
    if starts[-1] + length < samples:
        starts.append(samples - length)
    return starts


def windows(wave: np.ndarray, length: int, overlap: float = OVERLAP) -> tuple[np.ndarray, list[int]]:
    """[windows, length] array; audio shorter than one window is repeated to fill it, as in training."""
    if wave.size <= length:
        return fixed_clip(wave, length)[None], [0]
    starts = window_starts(wave.size, length, overlap)
    return np.stack([wave[start : start + length] for start in starts]), starts


def score_long(
    logits_of: Callable[[np.ndarray], np.ndarray],
    wave: np.ndarray,
    sample_rate: int,
    clip_seconds: float,
    probability_of: Callable[[np.ndarray], np.ndarray] | None = None,
) -> dict:
    """`logits_of` maps a [n, samples] float32 batch to n logits, from PyTorch or ONNX Runtime alike.

    Returns logits aggregated both ways and a per-window timeline. Pass
    `probability_of` (calibration) to add probabilities to the timeline.
    """
    length = int(clip_seconds * sample_rate)
    batch, starts = windows(wave.astype(np.float32), length)
    logits = np.concatenate([logits_of(batch[index : index + BATCH]) for index in range(0, len(batch), BATCH)])
    duration = wave.size / sample_rate
    timeline = []
    for start, logit in zip(starts, logits):
        entry = {
            "start": round(start / sample_rate, 3),
            "end": round(min(start + length, wave.size) / sample_rate, 3) if wave.size > length else round(duration, 3),
            "logit": float(logit),
        }
        if probability_of is not None:
            entry["probability"] = float(probability_of(np.array([logit]))[0])
        timeline.append(entry)
    return {
        "duration": round(duration, 3),
        "mean_logit": float(logits.mean()),
        "max_logit": float(logits.max()),
        "windows": timeline,
    }
