from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image
from skimage import data

from ml.common.manifest import FAKE, REAL, REAL_GENERATOR, Record
from ml.common.manifest import records_to_frame
from ml.common.splits import SplitConfig, assign_splits
from ml.image import datasets, preprocess
from ml.image.preprocess import Detection

CONFIG = {"size": 224, "frames_per_video": 3, "face_margin": 1.3, "face_threshold": 0.9}


def write_image(path: Path, size=(300, 200)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = np.random.default_rng(0).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    Image.fromarray(pixels).save(path)
    return path


def write_video(path: Path, frames: int = 12) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 64))
    for index in range(frames):
        writer.write(np.full((64, 64, 3), index * 20, dtype=np.uint8))
    writer.release()
    return path


def test_normalize_uses_imagenet_statistics():
    pixel = np.array([[[124, 116, 104]]], dtype=np.uint8)
    assert np.abs(preprocess.normalize(pixel)).max() < 0.02


def test_resize_square_handles_wide_and_tall_images():
    assert preprocess.resize_square(np.zeros((100, 400, 3), np.uint8), 224).shape == (224, 224, 3)
    assert preprocess.resize_square(np.zeros((900, 300, 3), np.uint8), 224).shape == (224, 224, 3)


def test_crop_face_stays_inside_frame_and_stays_square():
    rgb = np.zeros((200, 300, 3), np.uint8)
    assert preprocess.crop_face(rgb, (250, 150, 299, 199), 1.3).shape[:2] == (63, 63)
    assert preprocess.crop_face(rgb, (0, 0, 280, 190), 1.3).shape[:2] == (200, 200)


def test_sample_frames_returns_evenly_spaced_rgb(tmp_path):
    frames = list(preprocess.sample_frames(write_video(tmp_path / "v.mp4"), 3))
    assert len(frames) == 3
    assert frames[0].mean() < frames[2].mean()


def test_prepare_resizes_whole_images_to_png(tmp_path):
    source = write_image(tmp_path / "raw" / "a.jpg")
    record = Record(str(source), REAL, "toy", REAL_GENERATOR, "a", domain="whole_image")
    kept, stats = preprocess.prepare([record], tmp_path / "out", CONFIG)
    assert stats["output_records"] == 1
    with Image.open(tmp_path / "out" / kept[0].path) as image:
        assert image.size == (224, 224)
        assert image.format == "PNG"


def test_prepare_expands_video_to_frames_in_one_group(tmp_path):
    source = write_video(tmp_path / "raw" / "v.mp4")
    record = Record(str(source), FAKE, "toy", "gen", "person", domain="face")
    detector = lambda rgb: [Detection((5.0, 5.0, 50.0, 50.0), 0.99)]  # noqa: E731
    kept, _ = preprocess.prepare([record], tmp_path / "out", CONFIG, detector=detector)
    assert len(kept) == 3
    assert {item.group_id for item in kept} == {"person"}
    assert len({item.path for item in kept}) == 3


def test_prepare_uses_the_largest_face(tmp_path):
    pixels = np.zeros((200, 200, 3), np.uint8)
    pixels[100:190, 100:190] = 255
    source = tmp_path / "raw.png"
    Image.fromarray(pixels).save(source)
    record = Record(str(source), REAL, "toy", REAL_GENERATOR, "a", domain="face")
    faces = [Detection((0.0, 0.0, 20.0, 20.0), 0.99), Detection((100.0, 100.0, 190.0, 190.0), 0.95)]
    kept, _ = preprocess.prepare([record], tmp_path / "out", CONFIG, detector=lambda rgb: faces)
    with Image.open(tmp_path / "out" / kept[0].path) as image:
        assert np.asarray(image).mean() > 150


def test_prepare_counts_frames_without_faces_and_bad_files(tmp_path):
    good = write_image(tmp_path / "raw" / "a.png")
    bad = tmp_path / "raw" / "b.png"
    bad.write_bytes(b"not an image")
    records = [
        Record(str(good), REAL, "toy", REAL_GENERATOR, "a", domain="face"),
        Record(str(bad), REAL, "toy", REAL_GENERATOR, "b", domain="whole_image"),
    ]
    kept, stats = preprocess.prepare(records, tmp_path / "out", CONFIG, detector=lambda rgb: [])
    assert kept == []
    assert stats["dropped_no_face"] == 1
    assert stats["dropped_unreadable"] == 1


def make_faceforensics(root: Path) -> None:
    for name in ("000", "003", "004", "005"):
        write_video(root / "original_sequences" / "youtube" / "c23" / "videos" / f"{name}.mp4", 2)
    for method in ("Deepfakes", "NeuralTextures"):
        for name in ("000_003", "003_000", "004_005", "005_004"):
            write_video(root / "manipulated_sequences" / method / "c23" / "videos" / f"{name}.mp4", 2)


def test_faceforensics_links_both_people_in_a_swap(tmp_path):
    make_faceforensics(tmp_path)
    rows = list(datasets.faceforensics_records(tmp_path, {}))
    groups = {Path(r.path).stem: r.group_id for r in rows}
    assert groups["000"] == groups["003"] == groups["000_003"] == groups["003_000"]
    assert groups["004"] == groups["005_004"]
    assert groups["000"] != groups["004"]
    assert {r.generator_name for r in rows} == {REAL_GENERATOR, "Deepfakes", "NeuralTextures"}


def test_faceforensics_identity_never_crosses_splits(tmp_path):
    make_faceforensics(tmp_path)
    rows = [
        Record(r.path, r.label, r.source_dataset, r.generator_name, f"{r.group_id}", content_hash=r.path)
        for r in datasets.faceforensics_records(tmp_path, {})
    ]
    pairs = {r.group_id for r in rows}
    assert len(pairs) == 2
    with pytest.raises(ValueError, match="groups"):
        assign_splits(records_to_frame(rows), SplitConfig())


def test_celebdf_reads_three_folders(tmp_path):
    write_video(tmp_path / "Celeb-real" / "id0_0000.mp4", 2)
    write_video(tmp_path / "YouTube-real" / "00001.mp4", 2)
    write_video(tmp_path / "Celeb-synthesis" / "id0_id16_0000.mp4", 2)
    rows = list(datasets.celebdf_records(tmp_path, {}))
    assert [(r.label, r.generator_name, r.group_id) for r in rows] == [
        (REAL, REAL_GENERATOR, "id0"),
        (REAL, REAL_GENERATOR, "00001"),
        (FAKE, "celeb_df_v2", "id0"),
    ]


def test_genimage_finds_nested_folders(tmp_path):
    write_image(tmp_path / "Midjourney" / "imagenet_midjourney" / "train" / "ai" / "a.png")
    write_image(tmp_path / "Midjourney" / "imagenet_midjourney" / "train" / "nature" / "b.jpg")
    write_image(tmp_path / "BigGAN" / "val" / "ai" / "c.png")
    rows = list(datasets.genimage_records(tmp_path, {}))
    assert sorted((r.generator_name, r.label) for r in rows) == [
        ("biggan", FAKE), ("midjourney", FAKE), (REAL_GENERATOR, REAL),
    ]


@pytest.mark.network
def test_retinaface_finds_the_astronaut_face_and_crops_it():
    detector = preprocess.retinaface_detector(0.9)
    rgb = data.astronaut()
    faces = detector(rgb)
    assert len(faces) == 1
    left, top, right, bottom = faces[0].box
    # The face in this standard test image sits near the top centre.
    assert 150 < (left + right) / 2 < 300
    assert 40 < (top + bottom) / 2 < 180
    crop = preprocess.crop_face(rgb, faces[0].box, 1.3)
    assert preprocess.resize_square(crop, 224).shape == (224, 224, 3)
