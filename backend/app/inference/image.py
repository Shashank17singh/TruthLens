"""Image inference and explainability pipeline."""

import io
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

from backend.app.inference.calibration import calibrated_probability
from backend.app.inference.calibration import verdict as get_verdict
from backend.app.inference.registry import registry
from ml.image.explain import describe, explain
from ml.image.preprocess import resize_square


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def compute_dct_spectrum(image_rgb: np.ndarray) -> np.ndarray:
    """Compute 2D Discrete Cosine Transform magnitude spectrum."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    dct = cv2.dct(gray)
    magnitude = np.log(np.abs(dct) + 1.0)
    norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return norm.astype(np.uint8)


def generate_spectrum_png(spectrum: np.ndarray) -> bytes:
    """Render spectrum as a colored PNG using colormap."""
    colored = cv2.applyColorMap(spectrum, cv2.COLORMAP_VIRIDIS)
    _, buffer = cv2.imencode(".png", colored)
    return buffer.tobytes()


def generate_heatmap_overlay_png(image_rgb: np.ndarray, heatmap: np.ndarray) -> bytes:
    """Overlay a heatmap [0, 1] on an RGB image and encode as PNG."""
    heatmap_uint8 = np.uint8(255 * np.clip(heatmap, 0, 1))
    colored_cam = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_cam = cv2.cvtColor(colored_cam, cv2.COLOR_BGR2RGB)
    overlay = np.float32(colored_cam) / 255 + np.float32(image_rgb) / 255
    overlay = overlay / np.max(overlay)
    overlay_uint8 = np.uint8(255 * overlay)
    pil_img = Image.fromarray(overlay_uint8)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def detect_face_and_crop(pil_img: Image.Image) -> tuple[Image.Image, bool, list[dict[str, Any]]]:
    """Detect face if available, otherwise return original image."""
    # Attempt RetinaFace through batch_face or OpenCV
    try:
        from batch_face import RetinaFace

        detector = RetinaFace(gpu_id=-1)
        np_img = np.array(pil_img.convert("RGB"))
        faces = detector(np_img, cv=False)
        if faces and len(faces) > 0:
            box, landmarks, score = faces[0]
            left, top, right, bottom = map(int, box)
            # Add small margin
            w, h = right - left, bottom - top
            margin_x, margin_y = int(w * 0.1), int(h * 0.1)
            img_w, img_h = pil_img.size
            crop_box = (
                max(0, left - margin_x),
                max(0, top - margin_y),
                min(img_w, right + margin_x),
                min(img_h, bottom + margin_y),
            )
            cropped = pil_img.crop(crop_box)
            regions = [{"box": [left, top, right, bottom], "score": float(score), "part": "face"}]
            return cropped, True, regions
    except Exception:
        pass

    return pil_img, False, []


def to_original_box(box: tuple[int, int, int, int], original_size: tuple[int, int], size: int) -> list[int]:
    """Map a box in model-input pixels back through resize_square's scale and centre crop."""
    width, height = original_size
    scale = size / min(width, height)
    offset_x = (max(size, round(width * scale)) - size) // 2
    offset_y = (max(size, round(height * scale)) - size) // 2
    left, top, right, bottom = box
    return [
        round((left + offset_x) / scale),
        round((top + offset_y) / scale),
        round((right + offset_x) / scale),
        round((bottom + offset_y) / scale),
    ]


def model_input(pil_img: Image.Image, info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Prepare an image the way training did: short side to size, centre crop, then normalize."""
    square = resize_square(np.array(pil_img.convert("RGB")), int(info.get("image_size", 224)))
    mean = np.array(info.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.array(info.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
    normalized = (square.astype(np.float32) / 255.0 - mean) / std
    return square, normalized.transpose(2, 0, 1)[np.newaxis].astype(np.float32)


def run_image_inference(image_bytes: bytes) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Run ONNX inference and explainability on image bytes.

    Returns (analysis_result_dict, artefact_bytes_dict).
    """
    start_time = time.perf_counter()
    session, info = registry.get_image_session()

    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    # Faces are reported, not cropped: the served model was trained on whole GenImage images.
    _, face_detected, regions = detect_face_and_crop(pil_img)
    square, tensor = model_input(pil_img, info)

    # Run ONNX inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: tensor.astype(np.float32)})
    raw_logit = float(outputs[0].flatten()[0])

    probability = calibrated_probability(raw_logit, info)

    threshold = float(info.get("decision_threshold", 0.50))
    verdict = get_verdict(probability, threshold)
    verdict_label = {"likely_fake": "Likely AI-generated", "likely_real": "Likely real", "uncertain": "Uncertain"}[
        verdict
    ]

    # Confidence band (bootstrap or variance heuristic)
    delta = min(0.08, max(0.02, 0.1 * abs(0.5 - probability)))
    conf_low = max(0.0, round(probability - delta, 3))
    conf_high = min(1.0, round(probability + delta, 3))

    # Explainability artefacts. The spectrum is always real; the heatmap and regions
    # come only from Grad-CAM on the trained PyTorch checkpoint, never from placeholders.
    spectrum_png = generate_spectrum_png(compute_dct_spectrum(square))
    artefacts = {"spectrum.png": spectrum_png}
    explainer = registry.get_image_explainer()
    if explainer is not None:
        model, checkpoint = explainer
        _, explanation = explain(model, checkpoint, np.array(pil_img))
        artefacts["heatmap.png"] = generate_heatmap_overlay_png(square, explanation.heatmap)
        regions = regions + [
            {"box": to_original_box(r.box, pil_img.size, square.shape[0]), "score": r.score, "part": r.part}
            for r in explanation.regions
        ]
        explanation_text = describe([r.part for r in explanation.regions], leans_fake=verdict == "likely_fake")
    else:
        explanation_text = f"{verdict_label}. No heatmap: this deployment has no PyTorch checkpoint for Grad-CAM."

    latency_ms = int((time.perf_counter() - start_time) * 1000)

    result_data = {
        "model_name": info.get("name", "image-convnext-tiny-freq"),
        "model_version": info.get("version", "1.0.0"),
        "probability": round(probability, 6),
        "decision_threshold": threshold,
        "verdict": verdict,
        "verdict_label": verdict_label,
        "confidence_band": {"low": conf_low, "high": conf_high},
        "explanation": explanation_text,
        "warnings": [],
        "disclaimer": info.get(
            "disclaimer",
            "Results are probabilistic and must not be used alone to make decisions about a person.",
        ),
        "latency_ms": latency_ms,
        "details": {
            "face_detected": face_detected,
            "regions": regions,
            "artefacts": {name.removesuffix(".png"): name for name in artefacts},
        },
    }

    return result_data, artefacts
