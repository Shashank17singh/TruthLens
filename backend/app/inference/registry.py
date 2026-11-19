"""Model registry: loads models once per worker process from models.yaml."""

import logging
from pathlib import Path
from typing import Any

import joblib
import onnxruntime
import yaml
from backend.app.core.config import settings

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, config_path: str | Path | None = None):
        self.config_path = Path(config_path or (settings.repo_root / settings.MODELS_YAML_PATH))
        self.config: dict[str, Any] = {}
        self._image_session: onnxruntime.InferenceSession | None = None
        self._audio_session: onnxruntime.InferenceSession | None = None
        self._text_meta: Any = None
        self._text_pair: Any = None
        self.load_config()

    def load_config(self) -> None:
        """Load and parse models.yaml."""
        if not self.config_path.exists():
            logger.warning("models.yaml not found at %s. Using default configuration.", self.config_path)
            self.config = {
                "version": "1.0.0",
                "models": {
                    "image": {
                        "name": "image-convnext-tiny-freq",
                        "version": "1.0.0",
                        "onnx_file": "checkpoints/onnx_benchmark/bench-convnext_tiny-freq.onnx",
                        "decision_threshold": 0.50,
                    },
                    "audio": {
                        "name": "audio-lcnn-lfcc",
                        "version": "1.0.0",
                        "onnx_file": "checkpoints/onnx_benchmark/bench-audio-lcnn-lfcc.onnx",
                        "decision_threshold": 0.50,
                    },
                    "text": {
                        "name": "text-stats-ensemble",
                        "version": "1.0.0",
                        "meta_file": "checkpoints/text/meta.joblib",
                        "decision_threshold": 0.50,
                    },
                },
            }
            return

        with open(self.config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file) or {}

    def get_model_info(self, modality: str) -> dict[str, Any]:
        return self.config.get("models", {}).get(modality, {})

    def get_image_session(self) -> tuple[onnxruntime.InferenceSession, dict[str, Any]]:
        info = self.get_model_info("image")
        if self._image_session is None:
            onnx_rel = info.get("onnx_file", "checkpoints/onnx_benchmark/bench-convnext_tiny-freq.onnx")
            onnx_path = settings.repo_root / onnx_rel
            if not onnx_path.exists():
                raise FileNotFoundError(f"Image ONNX model not found: {onnx_path}")
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = 2
            self._image_session = onnxruntime.InferenceSession(
                str(onnx_path), options, providers=["CPUExecutionProvider"]
            )
            logger.info("Loaded Image ONNX session from %s", onnx_path)
        return self._image_session, info

    def get_audio_session(self) -> tuple[onnxruntime.InferenceSession, dict[str, Any]]:
        info = self.get_model_info("audio")
        if self._audio_session is None:
            onnx_rel = info.get("onnx_file", "checkpoints/onnx_benchmark/bench-audio-lcnn-lfcc.onnx")
            onnx_path = settings.repo_root / onnx_rel
            if not onnx_path.exists():
                raise FileNotFoundError(f"Audio ONNX model not found: {onnx_path}")
            options = onnxruntime.SessionOptions()
            options.intra_op_num_threads = 2
            self._audio_session = onnxruntime.InferenceSession(
                str(onnx_path), options, providers=["CPUExecutionProvider"]
            )
            logger.info("Loaded Audio ONNX session from %s", onnx_path)
        return self._audio_session, info

    def get_text_components(self) -> tuple[Any, Any, dict[str, Any]]:
        info = self.get_model_info("text")
        if self._text_meta is None:
            meta_rel = info.get("meta_file", "checkpoints/text/meta.joblib")
            meta_path = settings.repo_root / meta_rel
            if not meta_path.exists():
                raise FileNotFoundError(f"Text meta pipeline not found: {meta_path}")
            self._text_meta = joblib.load(meta_path)
            logger.info("Loaded Text meta pipeline from %s", meta_path)

        if self._text_pair is None:
            from ml.text.features import ModelPair
            self._text_pair = ModelPair.pretrained(device="cpu")
            logger.info("Loaded Text GPT-2 / DistilGPT-2 model pair")

        return self._text_meta, self._text_pair, info

    def get_catalog(self) -> list[dict[str, Any]]:
        catalog = []
        for modality, data in self.config.get("models", {}).items():
            catalog.append({
                "modality": modality,
                "name": data.get("name"),
                "version": data.get("version"),
                "decision_threshold": data.get("decision_threshold", 0.50),
            })
        return catalog


registry = ModelRegistry()
