"""Serving must prepare an image exactly as training did (ml/image/preprocess.py)."""

import numpy as np
from PIL import Image

from backend.app.inference.image import model_input
from ml.image.preprocess import resize_square


def test_model_input_matches_training_preprocessing():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(300, 450, 3), dtype=np.uint8)
    info = {"image_size": 224, "mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]}

    square, tensor = model_input(Image.fromarray(rgb), info)

    expected = resize_square(rgb, 224)
    assert np.array_equal(square, expected)
    normalized = (expected.astype(np.float32) / 255.0 - 0.5) / 0.25
    assert tensor.shape == (1, 3, 224, 224)
    assert np.allclose(tensor[0], normalized.transpose(2, 0, 1))


class _FakeSession:
    def get_inputs(self):
        return [type("Input", (), {"name": "image"})()]

    def run(self, outputs, feeds):
        return [np.array([[2.0]], dtype=np.float32)]


def test_without_a_pytorch_checkpoint_nothing_is_invented(monkeypatch):
    import io

    from backend.app.inference import image as image_module

    info = {"onnx_file": "x.onnx", "pytorch_checkpoint": None, "decision_threshold": 0.5}
    monkeypatch.setattr(image_module.registry, "get_image_session", lambda: (_FakeSession(), info))
    monkeypatch.setattr(image_module.registry, "get_image_explainer", lambda: None)
    monkeypatch.setattr(image_module, "detect_face_and_crop", lambda img: (img, False, []))
    buffer = io.BytesIO()
    Image.fromarray(np.full((64, 96, 3), 128, np.uint8)).save(buffer, format="PNG")

    result, artefacts = image_module.run_image_inference(buffer.getvalue())

    assert result["details"]["regions"] == []
    assert "heatmap.png" not in artefacts
    assert "heatmap" not in result["details"]["artefacts"]
    assert "spectrum.png" in artefacts


def test_region_boxes_map_back_to_original_pixels():
    from backend.app.inference.image import to_original_box

    # 448x300 (w x h) scales by 224/300; the crop drops (334 - 224) // 2 = 55 columns on the left.
    assert to_original_box((0, 0, 224, 224), (448, 300), 224) == [74, 0, 374, 300]
