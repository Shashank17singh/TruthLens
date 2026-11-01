"""Image preparation: frame sampling, face cropping, resizing and normalization."""

import hashlib
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from ml.common.checksums import hash_file
from ml.common.manifest import Record

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov"}


@dataclass(frozen=True)
class Detection:
    box: tuple[float, float, float, float]
    score: float


FaceDetector = Callable[[np.ndarray], list[Detection]]


def retinaface_detector(threshold: float) -> FaceDetector:
    """RetinaFace through batch-face. It runs on current PyTorch, unlike facenet-pytorch."""
    import torch
    from batch_face import RetinaFace

    model = RetinaFace(gpu_id=0 if torch.cuda.is_available() else -1, device="cuda" if torch.cuda.is_available() else "cpu")

    def detect(rgb: np.ndarray) -> list[Detection]:
        faces = model(rgb, cv=False, threshold=threshold)
        return [Detection(tuple(float(v) for v in box), float(score)) for box, _, score in faces]

    return detect


def normalize(rgb: np.ndarray) -> np.ndarray:
    """Scale a uint8 RGB image and apply ImageNet statistics, as the timm backbones expect."""
    return (rgb.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


def sample_frames(path: Path, count: int) -> Iterator[np.ndarray]:
    """Yield evenly spaced RGB frames from a video."""
    capture = cv2.VideoCapture(str(path))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        for index in np.linspace(0, max(total - 1, 0), num=count, dtype=int):
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if ok:
                yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        capture.release()


def crop_face(rgb: np.ndarray, box: tuple[float, float, float, float], margin: float) -> np.ndarray:
    """Crop a square around the face, shifted to stay inside the frame."""
    height, width = rgb.shape[:2]
    left, top, right, bottom = box
    side = min(int(max(right - left, bottom - top) * margin), height, width)
    x = int((left + right - side) / 2)
    y = int((top + bottom - side) / 2)
    x = min(max(x, 0), width - side)
    y = min(max(y, 0), height - side)
    return rgb[y : y + side, x : x + side]


def resize_square(rgb: np.ndarray, size: int) -> np.ndarray:
    """Scale the short side to `size` and centre-crop, which keeps the aspect ratio."""
    height, width = rgb.shape[:2]
    scale = size / min(height, width)
    resized = cv2.resize(
        rgb,
        (max(size, round(width * scale)), max(size, round(height * scale))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    top = (resized.shape[0] - size) // 2
    left = (resized.shape[1] - size) // 2
    return resized[top : top + size, left : left + size]


def _frames_of(record: Record, count: int) -> Iterator[tuple[int, np.ndarray]]:
    path = Path(record.path)
    if path.suffix.lower() in VIDEO_SUFFIXES:
        yield from enumerate(sample_frames(path, count))
        return
    with Image.open(path) as image:
        yield 0, np.asarray(image.convert("RGB"))


def _save(record: Record, frame_index: int, rgb: np.ndarray, output_dir: Path) -> Record:
    name = hashlib.sha256(f"{record.source_dataset}/{record.path}/{frame_index}".encode()).hexdigest()[:24]
    relative = f"image/{name[:2]}/{name}.png"
    target = output_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    # PNG adds no compression of its own, so no new codec trace separates the classes.
    Image.fromarray(rgb).save(target, compress_level=6)
    return replace(record, path=relative, content_hash=hash_file(target))


def prepare(
    records: Sequence[Record],
    output_dir: Path,
    config: dict,
    workers: int = 1,
    detector: FaceDetector | None = None,
) -> tuple[list[Record], dict[str, int]]:
    """Face datasets are cropped, and a frame without a face is dropped and counted."""
    stats: Counter = Counter(input_records=len(records))
    size = config["size"]
    if detector is None and any(record.domain == "face" for record in records):
        detector = retinaface_detector(config["face_threshold"])

    prepared: list[Record] = []
    for record in records:
        try:
            frames = list(_frames_of(record, config["frames_per_video"]))
        except (UnidentifiedImageError, OSError):
            stats["dropped_unreadable"] += 1
            continue
        if not frames:
            stats["dropped_unreadable"] += 1
        for frame_index, rgb in frames:
            if record.domain == "face":
                faces = detector(rgb)
                if not faces:
                    stats["dropped_no_face"] += 1
                    continue
                largest = max(faces, key=lambda f: (f.box[2] - f.box[0]) * (f.box[3] - f.box[1]))
                rgb = crop_face(rgb, largest.box, config["face_margin"])
            prepared.append(_save(record, frame_index, resize_square(rgb, size), output_dir))

    stats["output_records"] = len(prepared)
    return prepared, dict(stats)
