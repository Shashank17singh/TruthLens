"""Audio model, training, evaluation, export and saliency tests on generated audio.

Fake clips carry a faint high tone, a stand-in for the narrow-band trace a
vocoder can leave. Training on it shows the pipeline learns, and the numbers
here say nothing about real-world accuracy.
"""

import io
import json
import shutil
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import pytest
import soundfile
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from scipy.stats import norm

from ml.audio import augment, evaluate, explain, export, inference
from ml.audio.features import (
    Cqt,
    LogMel,
    PowerSpectrogram,
    build_frontend,
    linear_filterbank,
)
from ml.audio.model import (
    Lcnn,
    SslDetector,
    build_model,
    load_checkpoint,
    unfreeze_schedule,
)
from ml.audio.train import run_name, train
from ml.common import metrics
from ml.common.manifest import (
    FAKE,
    REAL,
    REAL_GENERATOR,
    Record,
    records_to_frame,
    write_manifest,
)
from ml.common.splits import SplitConfig, assign_splits

CONFIG_DIR = str(Path(__file__).parents[1] / "configs")
RATE = 16000
CLIP_SECONDS = 1
TONE_HZ = 6000
HAS_FFMPEG = shutil.which("ffmpeg") is not None
TINY_SSL = {
    "hidden_size": 32, "num_hidden_layers": 2, "num_attention_heads": 2, "intermediate_size": 64,
    "num_feat_extract_layers": 2, "conv_dim": [16, 16], "conv_stride": [5, 4], "conv_kernel": [10, 8],
    "num_conv_pos_embeddings": 16, "num_conv_pos_embedding_groups": 2,
}


def experiment_config(tmp_path: Path, *overrides: str) -> dict:
    values = {
        "model": "lcnn", "feature": "mel", "augment": "none", "tracker": "none", "num_workers": "0", "device": "cpu",
        "clip_seconds": str(CLIP_SECONDS), "epochs": "6", "patience": "6", "model.batch_size": "16",
        "model.learning_rate": "0.003", "label_smoothing": "0.0",
        "manifest": (tmp_path / "toy.csv").as_posix(), "data_root": (tmp_path / "data").as_posix(),
        "checkpoint_dir": (tmp_path / "checkpoints").as_posix(),
    }
    values.update(override.split("=", 1) for override in overrides)
    with initialize_config_dir(CONFIG_DIR, version_base=None):
        config = compose("train", overrides=[f"{key}={value}" for key, value in values.items()])
    return OmegaConf.to_container(config, resolve=True)


def ssl_config(tmp_path: Path, model_type: str = "wav2vec2") -> dict:
    config = experiment_config(tmp_path, "model=wav2vec2", "model.pretrained=false", "model.freeze_epochs=1", "model.unfreeze_layers=1")
    config["model"]["config_overrides"] = {"model_type": model_type, **TINY_SSL}
    return config


def speech(rng: np.random.Generator, seconds: float, tone: float = 0.0) -> np.ndarray:
    clip = export.generated_speech(1, int(seconds * RATE), RATE, int(rng.integers(1 << 30)))[0]
    return (clip + tone * np.sin(2 * np.pi * TONE_HZ * np.arange(clip.size) / RATE)).astype(np.float32)


@pytest.fixture(scope="module")
def toy_data(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("toy_audio")
    rng = np.random.default_rng(0)
    folder = tmp_path / "data" / "toy"
    folder.mkdir(parents=True)
    records = []
    for index in range(60):
        for label, generator, tone in ((REAL, REAL_GENERATOR, 0.0), (FAKE, "seen_tts", 0.05)):
            name = f"{generator}-{index}.flac"
            soundfile.write(folder / name, speech(rng, rng.uniform(0.6, 2.5), tone), RATE)
            records.append(Record(name, label, "toy", generator, f"speaker{index}", content_hash=name))
    for index in range(15):
        name = f"new_tts-{index}.flac"
        soundfile.write(folder / name, speech(rng, 1.5, 0.05), RATE)
        records.append(Record(name, FAKE, "toy", "new_tts", f"new{index}", content_hash=name))
    result = assign_splits(records_to_frame(records), SplitConfig(heldout_generators=frozenset({"new_tts"})))
    write_manifest(result.frame, tmp_path / "toy.csv")
    return tmp_path


# Features


def test_power_spectrogram_matches_librosa():
    wave = np.random.default_rng(0).standard_normal(RATE).astype(np.float32) * 0.1
    expected = np.abs(librosa.stft(wave, n_fft=512, hop_length=160, win_length=400, center=True, pad_mode="reflect")) ** 2
    actual = PowerSpectrogram()(torch.from_numpy(wave)[None])[0].numpy()
    assert actual.shape == expected.shape
    assert np.abs(actual - expected).max() / expected.max() < 1e-4


@pytest.mark.parametrize("kind,bins", [("mel", 128), ("lfcc", 60)])
def test_frontends_give_one_channel_images(kind, bins):
    features = build_frontend(kind, RATE)(torch.zeros(3, 2 * RATE) + 0.01)
    assert features.shape[:3] == (3, 1, bins) and torch.isfinite(features).all()


def test_a_tone_lights_up_the_right_mel_and_cqt_bins():
    tone = torch.from_numpy(np.sin(2 * np.pi * 1000 * np.arange(RATE) / RATE).astype(np.float32))[None]
    mel_peak = int(LogMel(RATE)(tone)[0, 0].mean(dim=1).argmax())
    mel_centres = librosa.mel_frequencies(n_mels=130, fmin=20.0, fmax=RATE / 2)[1:-1]
    assert abs(mel_centres[mel_peak] - 1000) < 60
    cqt = Cqt(RATE)
    cqt_peak = int(cqt(tone)[0, 0].mean(dim=1).argmax())
    assert abs(librosa.cqt_frequencies(n_bins=cqt.bins, fmin=50.0)[cqt_peak] - 1000) < 1000 * 0.06


def test_linear_filters_cover_the_band_evenly():
    """Between the first and last filter centres, overlapping triangles sum to one."""
    filters = 70
    total = linear_filterbank(RATE, 512, filters).sum(axis=0)
    bin_hz = np.linspace(0, RATE / 2, 257)
    centres = np.linspace(0, RATE / 2, filters + 2)[1:-1]
    inside = (bin_hz >= centres[0]) & (bin_hz <= centres[-1])
    assert np.allclose(total[inside], 1.0, atol=1e-5)


# Augmentations


def test_speed_perturbation_rescales_length():
    wave = np.zeros(RATE, dtype=np.float32)
    assert abs(augment.change_speed(wave, 1.1, RATE).size - RATE / 1.1) <= 2
    assert abs(augment.change_speed(wave, 0.9, RATE).size - RATE / 0.9) <= 2


@pytest.mark.parametrize("snr", [10.0, 20.0])
def test_noise_is_added_at_the_requested_snr(snr):
    rng = np.random.default_rng(1)
    wave = speech(rng, 2.0)
    noisy = augment.add_noise(wave, snr, rng)
    measured = 10 * np.log10(np.mean(wave**2) / np.mean((noisy - wave) ** 2))
    assert abs(measured - snr) < 0.2


def test_reverb_keeps_length_and_peak_and_smears_energy():
    rng = np.random.default_rng(2)
    click = np.zeros(RATE, dtype=np.float32)
    click[100] = 1.0
    wet = augment.add_reverb(click, 0.5, RATE, rng)
    assert wet.size == click.size and np.isclose(np.abs(wet).max(), 1.0)
    assert np.mean(np.abs(wet[2000:4000])) > 0  # the tail reaches well after the click


@pytest.mark.skipif(not HAS_FFMPEG, reason="needs ffmpeg")
@pytest.mark.parametrize("codec", ["mp3", "opus"])
def test_codec_roundtrip_keeps_length_and_alters_audio(codec):
    wave = speech(np.random.default_rng(3), 2.0)
    decoded = augment.codec_roundtrip(wave, codec, 64, RATE)
    assert decoded.size == wave.size
    # Aligned: a shift of even 1 ms would make the error larger than the signal.
    assert 0 < np.mean((decoded - wave) ** 2) < 0.1 * np.mean(wave**2)


@pytest.mark.skipif(not HAS_FFMPEG, reason="needs ffmpeg")
def test_mp3_is_encoded_at_64_kbps():
    pcm = io.BytesIO()
    soundfile.write(pcm, speech(np.random.default_rng(4), 10.0), RATE, format="WAV", subtype="PCM_16")
    encoded = augment._ffmpeg(["-f", "wav", "-i", "pipe:0", *augment.CODEC_ARGUMENTS["mp3"], "-b:a", "64k", "pipe:1"], pcm.getvalue())
    assert abs(len(encoded) * 8 / 10 / 1000 - 64) < 4


def test_disabled_augmentations_leave_audio_untouched(tmp_path):
    config = experiment_config(tmp_path, "augment=none")
    augmenter = augment.Augmenter(config["augment"], RATE, 0)
    wave = speech(np.random.default_rng(5), 1.0)
    assert np.array_equal(augmenter(augmenter.before_crop(wave)), wave)


# Long audio


def test_windows_overlap_by_half_and_cover_the_tail():
    starts = inference.window_starts(30 * RATE, 4 * RATE)
    assert starts[:3] == [0, 2 * RATE, 4 * RATE]
    assert starts[-1] + 4 * RATE == 30 * RATE and len(starts) == 14
    assert inference.window_starts(9 * RATE, 4 * RATE)[-1] == 5 * RATE


def test_short_audio_becomes_one_full_window():
    batch, starts = inference.windows(np.ones(RATE, dtype=np.float32), 4 * RATE)
    assert batch.shape == (1, 4 * RATE) and starts == [0]


def test_long_scores_report_mean_max_and_a_timeline():
    wave = np.zeros(10 * RATE, dtype=np.float32)
    wave[6 * RATE :] = 1.0  # the "fake" second half

    def logits_of(batch):
        return batch.mean(axis=1) * 4 - 2

    result = inference.score_long(logits_of, wave, RATE, 4, probability_of=metrics.sigmoid)
    assert [window["start"] for window in result["windows"]] == [0.0, 2.0, 4.0, 6.0]
    assert result["max_logit"] == pytest.approx(2.0) and result["mean_logit"] < result["max_logit"]
    assert result["windows"][-1]["end"] == 10.0 and "probability" in result["windows"][0]


# Metrics


def test_eer_is_zero_when_separable_and_matches_theory_for_gaussians():
    assert metrics.equal_error_rate(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))["eer"] == pytest.approx(0.0)
    rng = np.random.default_rng(0)
    scores = np.r_[rng.normal(0, 1, 20000), rng.normal(2, 1, 20000)]
    labels = np.r_[np.zeros(20000), np.ones(20000)]
    assert metrics.equal_error_rate(labels, scores)["eer"] == pytest.approx(norm.cdf(-1.0), abs=0.01)


ASV = {"p_fa_asv": 0.01, "p_miss_asv": 0.02, "p_miss_spoof_asv": 0.3}


def test_min_tdcf_is_zero_for_a_perfect_countermeasure_and_one_for_a_useless_one():
    labels = np.r_[np.zeros(100), np.ones(100)]
    assert metrics.min_tdcf(labels, np.r_[np.zeros(100), np.ones(100)], ASV) == pytest.approx(0.0)
    assert metrics.min_tdcf(labels, np.zeros(200), ASV) == pytest.approx(1.0)


def test_asv_error_rates_follow_the_official_threshold_rule():
    """The official DET threshold is the highest nontarget score here, and false alarms count `>=`.

    So one of the four nontargets is a false alarm even though the scores
    separate perfectly. That is the official package's behaviour, kept so the
    t-DCF matches published ASVspoof numbers.
    """
    keys = np.array(["target"] * 4 + ["nontarget"] * 4 + ["spoof"] * 2)
    scores = np.array([5, 6, 7, 8, 0, 1, 2, 3, 9, 1.0])
    assert metrics.asv_error_rates(keys, scores) == {"p_fa_asv": 0.25, "p_miss_asv": 0.0, "p_miss_spoof_asv": 0.5}


def test_per_attack_eer_compares_each_attack_with_all_real_clips():
    frame = pd.DataFrame({
        "label": [0, 0, 1, 1, 1, 1], "generator_name": ["real", "real", "A01", "A01", "A02", "A02"],
        "logit": [0.0, 0.1, 0.9, 0.8, 0.05, 0.02],
    })
    rows = {row["generator_name"]: row["eer"] for row in metrics.per_generator_eer(frame).to_dict("records")}
    assert rows["A01"] == pytest.approx(0.0) and rows["A02"] > 0.4


# Models


@pytest.mark.parametrize("model_name", ["resnet18", "lcnn"])
@pytest.mark.parametrize("feature", ["mel", "lfcc", "cqt"])
def test_spectrogram_models_return_one_logit_per_clip(tmp_path, model_name, feature):
    config = experiment_config(tmp_path, f"model={model_name}", f"feature={feature}", "model.pretrained=false")
    model = build_model(config).eval()
    with torch.no_grad():
        assert model(torch.randn(2, RATE) * 0.1).shape == (2,)


@pytest.mark.parametrize("model_type", ["wav2vec2", "wavlm"])
def test_ssl_model_trains_head_first_then_top_layers(tmp_path, model_type):
    config = ssl_config(tmp_path, model_type)
    model = build_model(config)
    assert isinstance(model, SslDetector)
    with torch.no_grad():
        assert model(torch.randn(2, RATE) * 0.1).shape == (2,)

    def trainable(module):
        return any(parameter.requires_grad for parameter in module.parameters())

    layers = model.backbone.encoder.layers
    assert trainable(model.head) and not trainable(layers[-1])
    unfreeze_schedule(config)(model, 1)
    assert trainable(layers[-1]) and not trainable(layers[0]) and not trainable(model.backbone.feature_extractor)


def test_lcnn_accepts_any_feature_size():
    assert Lcnn()(torch.randn(2, 1, 60, 101)).shape == (2, 1)
    assert Lcnn()(torch.randn(2, 1, 128, 401)).shape == (2, 1)


# Training, evaluation, export and saliency


@pytest.fixture(scope="module")
def trained(toy_data: Path) -> tuple[Path, Path]:
    config = experiment_config(toy_data)
    return train(config), toy_data


def test_training_learns_the_planted_tone(trained):
    best, _ = trained
    checkpoint = torch.load(best, weights_only=False)
    assert checkpoint["best_auc"] > 0.9


def test_checkpoint_name_carries_model_feature_and_version(trained, tmp_path):
    best, _ = trained
    assert best.name == "audio-lcnn-mel-none-1.0.0-best.pt"
    assert run_name(ssl_config(tmp_path)) == "audio-wav2vec2-wave-none-1.0.0"


def test_evaluation_reports_eer_both_aggregates_and_attacks(trained, tmp_path):
    best, _ = trained
    report = evaluate.evaluate(best, None, None, tmp_path / "results")
    for aggregate in ("mean", "max"):
        test = report[aggregate]["test"]
        assert {"eer", "accuracy", "f1", "roc_auc", "auc_interval", "ece_calibrated"} <= set(test)
        assert report[aggregate]["unseen"] is not None
        assert {row["generator_name"] for row in report[aggregate]["per_attack_eer"]} == {"seen_tts", "new_tts"}
    assert report["mean"]["test"]["eer"] < 0.2
    assert report["unseen_generators"] == ["new_tts"]
    for name in ("metrics.json", "roc.png", "confusion.png", "reliability.png", "attacks.png"):
        assert (tmp_path / "results" / name).exists()


def test_evaluation_adds_min_tdcf_with_asv_scores(trained, tmp_path):
    best, _ = trained
    scores = tmp_path / "asv.txt"
    lines = [f"spk target {value}" for value in np.linspace(3, 6, 20)] + [f"spk nontarget {value}" for value in np.linspace(-2, 3.5, 20)]
    lines += [f"spk spoof {value}" for value in np.linspace(0, 5, 20)]
    scores.write_text("\n".join(lines))
    report = evaluate.evaluate(best, None, None, tmp_path / "results", scores)
    assert 0.0 <= report["mean"]["test"]["min_tdcf"] <= 1.0


def test_onnx_export_matches_pytorch_and_times_a_30_second_clip(trained, tmp_path):
    best, _ = trained
    report = export.export_checkpoint(best, tmp_path / "onnx", thread_counts=(1,))
    assert report["format"] == "onnx" and report["max_output_difference"] <= export.TOLERANCE
    assert report["latency_30s"][0]["clip_seconds"] == 30 and report["latency_30s"][0]["median_ms"] > 0
    assert json.loads((tmp_path / "onnx" / f"{report['name']}.json").read_text())["sha256"] == report["sha256"]


def test_ssl_model_exports_in_some_format_that_matches(tmp_path):
    config = ssl_config(tmp_path)
    model = build_model(config).eval()
    clips = export.generated_speech(export.VERIFY_CLIPS, CLIP_SECONDS * RATE, RATE, 0)
    report = export.export_model(model, "tiny-ssl", config, clips, "generated", tmp_path / "onnx", (1,))
    assert report["format"] in ("onnx", "torchscript")
    assert report["max_output_difference"] <= export.TOLERANCE


def test_saliency_points_at_the_planted_tone(trained):
    best, _ = trained
    model, _ = load_checkpoint(best)
    clip = speech(np.random.default_rng(9), CLIP_SECONDS, 0.05)
    result = explain.saliency(model, clip)
    assert result["kind"] == "time-frequency" and result["map"].shape == result["features"].shape
    assert 0.0 <= result["map"].min() and np.isclose(result["map"].max(), 1.0)
    centres = librosa.mel_frequencies(n_mels=130, fmin=20.0, fmax=RATE / 2)[1:-1]
    tone_rows = np.abs(centres - TONE_HZ) < 400
    assert result["map"][tone_rows].mean() > 1.5 * result["map"].mean()


def test_ssl_saliency_is_a_time_profile(tmp_path):
    model = build_model(ssl_config(tmp_path))
    result = explain.saliency(model, speech(np.random.default_rng(10), CLIP_SECONDS))
    assert result["kind"] == "time" and result["map"].shape == (1, CLIP_SECONDS * RATE // 160)
