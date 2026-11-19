"""Unit tests for the multi-modal robustness and adversarial evaluation suite."""

import numpy as np
import pytest
import torch
from PIL import Image

from ml.eval.robustness import (
    AudioPerturbations,
    GeneralizationResult,
    ImagePerturbations,
    TextPerturbations,
    ToyImageDetector,
    compute_cross_dataset_generalization,
    fgsm_attack,
    pgd_attack,
    plot_adversarial_curves,
    plot_audio_robustness,
    plot_combined_robustness_grid,
    plot_cross_dataset_drop,
    plot_image_robustness,
    plot_text_robustness,
    run_adversarial_attack_benchmark,
    run_audio_perturbation_benchmark,
    run_image_perturbation_benchmark,
    run_text_perturbation_benchmark,
    train_adversarial_defense,
)


def test_image_perturbations():
    """Verify visual perturbation transformations preserve dimensions and type."""
    # Synthetic 224x224 RGB image
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")

    # JPEG 90, 70, 50
    for q in (90, 70, 50):
        out = ImagePerturbations.jpeg_compression(img, quality=q)
        assert out.size == (224, 224)
        assert out.mode == "RGB"

    # Resize 0.5x
    out_resize = ImagePerturbations.resize_downscale(img, scale=0.5)
    assert out_resize.size == (224, 224)

    # Blur
    out_blur = ImagePerturbations.gaussian_blur(img, radius=2.0)
    assert out_blur.size == (224, 224)

    # Screenshot
    out_screen = ImagePerturbations.screenshot_reencode(img)
    assert out_screen.size == (224, 224)

    # Social Media
    out_social = ImagePerturbations.social_media_compression(img)
    assert out_social.size == (224, 224)


def test_audio_perturbations():
    """Verify audio perturbation pipelines preserve waveform length and shape."""
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False, dtype=np.float32)
    # 440 Hz test sine wave
    wave = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)

    # MP3 128 kbps & 64 kbps
    for br in (128, 64):
        out_mp3 = AudioPerturbations.mp3_compression(wave, bitrate_kbps=br, sr=sr)
        assert len(out_mp3) == len(wave)
        assert out_mp3.dtype == np.float32

    # Additive Noise SNR 20 & 10 dB
    for snr in (20.0, 10.0):
        out_noise = AudioPerturbations.additive_noise(wave, snr_db=snr)
        assert len(out_noise) == len(wave)
        assert not np.array_equal(out_noise, wave)

    # Phone band filter
    out_phone = AudioPerturbations.phone_band_filter(wave, sr=sr)
    assert len(out_phone) == len(wave)
    # Attenuated higher harmonics
    assert np.max(np.abs(out_phone)) > 0


def test_text_perturbations():
    """Verify text perturbation operations."""
    sample = "In this study, we propose a robust model to detect synthetic text features."
    human_sample = "The authors gathered empirical observations across multiple experiments."

    # Typos
    typo_text = TextPerturbations.apply_typos(sample, typo_rate=0.08)
    assert len(typo_text) > 0
    assert typo_text != sample

    # Synonym swaps
    syn_text = TextPerturbations.apply_synonym_swaps(sample, swap_rate=0.5)
    assert len(syn_text) > 0

    # Paraphrase
    para_text = TextPerturbations.apply_paraphrase(sample)
    assert "Here, we introduce" in para_text or "To " in para_text or para_text != sample

    # Mixed paragraphs
    mixed_text = TextPerturbations.mix_paragraphs(sample, human_sample, mix_ratio=0.5)
    assert len(mixed_text) > 0


def test_adversarial_fgsm_and_pgd():
    """Verify FGSM and PGD white-box adversarial attacks and adversarial training."""
    torch.manual_seed(42)
    model = ToyImageDetector()
    model.eval()

    # Batch of 2 synthetic images [2, 3, 32, 32]
    images = torch.rand(2, 3, 32, 32)
    labels = torch.tensor([1, 0])
    epsilon = 8.0 / 255.0

    # 1. FGSM Attack
    adv_fgsm = fgsm_attack(model, images, labels, epsilon=epsilon)
    assert adv_fgsm.shape == images.shape
    assert (adv_fgsm >= 0.0).all() and (adv_fgsm <= 1.0).all()
    # Perturbation magnitude bounded by epsilon
    assert torch.max(torch.abs(adv_fgsm - images)) <= epsilon + 1e-5

    # 2. PGD Attack
    adv_pgd = pgd_attack(model, images, labels, epsilon=epsilon, num_steps=5)
    assert adv_pgd.shape == images.shape
    assert (adv_pgd >= 0.0).all() and (adv_pgd <= 1.0).all()
    assert torch.max(torch.abs(adv_pgd - images)) <= epsilon + 1e-5

    # 3. Adversarial Training Defense
    robust_model = train_adversarial_defense(model, images, labels, epochs=2, epsilon=epsilon)
    assert robust_model is not None


def test_cross_dataset_generalization():
    """Verify generalization results contain all 3 modalities and valid drops."""
    results = compute_cross_dataset_generalization()
    assert len(results) >= 6
    modalities = {r.modality for r in results}
    assert modalities == {"Image", "Audio", "Text"}

    for r in results:
        assert isinstance(r, GeneralizationResult)
        assert r.absolute_drop > 0  # Out-of-distribution drop is always positive
        assert 0.0 < r.relative_retention_pct < 100.0


def test_benchmark_dataframes_and_plotting(tmp_path):
    """Verify all benchmark runners generate non-empty DataFrames and valid plots."""
    df_img = run_image_perturbation_benchmark()
    assert len(df_img) >= 6
    plot_image_robustness(df_img, tmp_path / "img_test.png")
    assert (tmp_path / "img_test.png").exists()

    df_aud = run_audio_perturbation_benchmark()
    assert len(df_aud) >= 5
    plot_audio_robustness(df_aud, tmp_path / "aud_test.png")
    assert (tmp_path / "aud_test.png").exists()

    df_text = run_text_perturbation_benchmark()
    assert len(df_text) >= 8
    plot_text_robustness(df_text, tmp_path / "text_test.png")
    assert (tmp_path / "text_test.png").exists()

    df_adv = run_adversarial_attack_benchmark()
    assert len(df_adv) >= 5
    plot_adversarial_curves(df_adv, tmp_path / "adv_test.png")
    assert (tmp_path / "adv_test.png").exists()

    gen_results = compute_cross_dataset_generalization()
    plot_cross_dataset_drop(gen_results, tmp_path / "gen_test.png")
    assert (tmp_path / "gen_test.png").exists()

    plot_combined_robustness_grid(df_img, df_aud, df_text, df_adv, tmp_path / "grid_test.png")
    assert (tmp_path / "grid_test.png").exists()
