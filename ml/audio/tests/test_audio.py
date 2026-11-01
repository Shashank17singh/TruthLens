from pathlib import Path

import numpy as np
import pytest
import soundfile

from ml.audio import datasets, preprocess
from ml.common.manifest import FAKE, REAL, REAL_GENERATOR, UNKNOWN_GENERATOR, Record

CONFIG = {"sample_rate": 16000, "clip_seconds": 4, "min_seconds": 0.5, "trim_top_db": 40}


def tone(seconds: float, rate: int, amplitude: float = 0.5) -> np.ndarray:
    times = np.arange(int(seconds * rate)) / rate
    return (amplitude * np.sin(2 * np.pi * 440 * times)).astype(np.float32)


def write(path: Path, wave: np.ndarray, rate: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, wave, rate)
    return path


def real_record(path: Path) -> Record:
    return Record(str(path), REAL, "toy", REAL_GENERATOR, "speaker")


def test_load_mono_resamples_and_mixes_down(tmp_path):
    stereo = np.stack([tone(1.0, 44100), tone(1.0, 44100)], axis=1)
    path = write(tmp_path / "a.wav", stereo, 44100)
    wave = preprocess.load_mono(path, 16000)
    assert wave.ndim == 1
    assert wave.size == pytest.approx(16000, abs=2)
    assert wave.dtype == np.float32


def test_trim_silence_removes_quiet_edges():
    rate = 16000
    wave = np.concatenate([np.zeros(rate), tone(1.0, rate), np.zeros(rate)])
    trimmed = preprocess.trim_silence(wave, 40)
    # librosa trims in 2048-sample frames, so a little context remains on each side.
    assert rate <= trimmed.size <= rate + 2 * 2048


def test_fixed_clip_repeats_short_audio():
    clip = preprocess.fixed_clip(np.array([1.0, 2.0, 3.0]), 7)
    assert clip.tolist() == [1, 2, 3, 1, 2, 3, 1]


def test_fixed_clip_is_repeatable_without_rng_and_random_with_it():
    wave = np.arange(100, dtype=np.float32)
    assert preprocess.fixed_clip(wave, 10).tolist() == list(range(10))
    first = preprocess.fixed_clip(wave, 10, np.random.default_rng(3))
    second = preprocess.fixed_clip(wave, 10, np.random.default_rng(3))
    assert first.size == 10
    assert (first == second).all()


def test_fixed_clip_rejects_empty_audio():
    with pytest.raises(ValueError):
        preprocess.fixed_clip(np.array([]), 10)


def test_prepare_writes_16k_mono_flac(tmp_path):
    source = write(tmp_path / "raw" / "a.wav", tone(2.0, 22050), 22050)
    kept, stats = preprocess.prepare([real_record(source)], tmp_path / "out", CONFIG)
    assert stats["output_records"] == 1
    output = tmp_path / "out" / kept[0].path
    info = soundfile.info(output)
    assert (info.samplerate, info.channels, info.format) == (16000, 1, "FLAC")
    assert len(kept[0].content_hash) == 64


def test_prepare_in_worker_processes_matches_single_process(tmp_path):
    sources = [write(tmp_path / "raw" / f"{i}.wav", tone(1 + i / 10, 16000), 16000) for i in range(3)]
    records = [real_record(path) for path in sources]
    single, _ = preprocess.prepare(records, tmp_path / "one", CONFIG, workers=1)
    multi, _ = preprocess.prepare(records, tmp_path / "two", CONFIG, workers=2)
    assert [r.content_hash for r in single] == [r.content_hash for r in multi]


def test_prepare_drops_silent_and_corrupt_files(tmp_path):
    silent = write(tmp_path / "raw" / "silent.wav", np.zeros(16000, dtype=np.float32), 16000)
    corrupt = tmp_path / "raw" / "corrupt.wav"
    corrupt.write_bytes(b"not audio")
    kept, stats = preprocess.prepare([real_record(silent), real_record(corrupt)], tmp_path / "out", CONFIG)
    assert kept == []
    assert stats["dropped_too_short"] == 1
    assert stats["dropped_unreadable"] == 1


def test_asvspoof2019_maps_official_splits_and_speakers(tmp_path):
    protocols = tmp_path / "LA" / "ASVspoof2019_LA_cm_protocols"
    protocols.mkdir(parents=True)
    (protocols / "ASVspoof2019.LA.cm.train.trn.txt").write_text("LA_0079 LA_T_1 - - bonafide\n")
    (protocols / "ASVspoof2019.LA.cm.dev.trl.txt").write_text("LA_0069 LA_D_1 - A02 spoof\n")
    (protocols / "ASVspoof2019.LA.cm.eval.trl.txt").write_text("LA_0039 LA_E_1 - A13 spoof\n")
    rows = list(datasets.asvspoof2019_la_records(tmp_path, {}))
    assert [(r.split, r.label, r.generator_name, r.group_id) for r in rows] == [
        ("train", REAL, REAL_GENERATOR, "LA_0079"),
        ("val", FAKE, "A02", "LA_0069"),
        ("test", FAKE, "A13", "LA_0039"),
    ]
    assert rows[2].path.endswith("LA_E_1.flac")


def test_asvspoof2021_keeps_eval_phase_with_real_line_format(tmp_path):
    keys = tmp_path / "keys" / "DF" / "CM"
    keys.mkdir(parents=True)
    (keys / "trial_metadata.txt").write_text(
        "LA_0023 DF_E_2000011 nocodec asvspoof A14 spoof notrim progress traditional_vocoder - - - -\n"
        "TEF2 DF_E_2000013 low_m4a vcc2020 Task1-team20 spoof notrim eval neural_vocoder_nonautoregressive Task1 team20 FF E\n"
        "LA_0043 DF_E_2000026 mp3m4a asvspoof bonafide bonafide notrim eval bonafide - - - -\n"
    )
    rows = list(datasets.asvspoof2021_df_records(tmp_path, {}))
    assert [(r.generator_name, r.domain) for r in rows] == [
        ("Task1-team20", "low_m4a"),
        (REAL_GENERATOR, "mp3m4a"),
    ]


def test_wavefake_pairs_copies_with_their_real_utterance(tmp_path):
    write(tmp_path / "LJSpeech-1.1" / "wavs" / "LJ001-0001.wav", tone(1, 16000), 16000)
    write(tmp_path / "generated_audio" / "ljspeech_melgan" / "LJ001-0001_gen.wav", tone(1, 16000), 16000)
    write(tmp_path / "generated_audio" / "jsut_parallel_wavegan" / "BASIC0001_gen.wav", tone(1, 16000), 16000)
    rows = list(datasets.wavefake_records(tmp_path, {}))
    assert [(r.generator_name, r.group_id) for r in rows] == [
        (REAL_GENERATOR, "LJ001-0001"),
        ("melgan", "LJ001-0001"),
    ]


def test_in_the_wild_reads_meta_labels(tmp_path):
    (tmp_path / "meta.csv").write_text("file,speaker,label\n0.wav,Alec,bona-fide\n1.wav,Barack,spoof\n", encoding="utf-8")
    rows = list(datasets.in_the_wild_records(tmp_path, {}))
    assert [(r.label, r.generator_name, r.group_id) for r in rows] == [
        (REAL, REAL_GENERATOR, "Alec"),
        (FAKE, UNKNOWN_GENERATOR, "Barack"),
    ]
