"""Tests for model registry and configuration."""

from backend.app.inference.registry import registry


def test_model_registry_loads_config():
    assert "models" in registry.config
    assert "image" in registry.config["models"]
    assert "audio" in registry.config["models"]
    assert "text" in registry.config["models"]


def test_model_catalog():
    catalog = registry.get_catalog()
    assert len(catalog) == 3
    modalities = {item["modality"] for item in catalog}
    assert modalities == {"image", "audio", "text"}
