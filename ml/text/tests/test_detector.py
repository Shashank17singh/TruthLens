"""Text detector tests on tiny random models and a tokenizer built offline, so nothing is downloaded."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from sklearn.linear_model import LogisticRegression
from tokenizers import Tokenizer, models, pre_tokenizers, processors, trainers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
)

from ml.text import detector, ensemble, explain, robustness
from ml.text.datasets import prompt_group
from ml.text.features import (
    FEATURES,
    ModelPair,
    split_sentences,
    summarize,
    text_features,
)
from ml.text.model import build_model, encode_windows, load_checkpoint, score_text
from ml.text.train import train

CONFIG_DIR = str(Path(__file__).resolve().parents[1] / "configs")
WORDS = ["the", "a", "of", "to", "and", "in", "is", "was", "for", "on", "it", "with", "as", "at", "by", "this", "from", "that", "river", "city", "music", "science", "market", "game", "story"]
MARKER = "zephyr"
TINY_ENCODER = {
    "model_type": "roberta", "vocab_size": 400, "hidden_size": 32, "num_hidden_layers": 1, "num_attention_heads": 2,
    "intermediate_size": 64, "max_position_embeddings": 80, "pad_token_id": 1, "bos_token_id": 0, "eos_token_id": 2,
    "hidden_dropout_prob": 0.0, "attention_probs_dropout_prob": 0.0,
}


def toy_text(rng: np.random.Generator, ai: bool, sentences: int = 4) -> str:
    """AI texts carry a marker word in every sentence; human texts never do."""
    out = []
    for _ in range(sentences):
        words = list(rng.choice(WORDS, int(rng.integers(6, 10))))
        if ai:
            words.insert(int(rng.integers(len(words))), MARKER)
        out.append(" ".join(words).capitalize() + ".")
    return " ".join(out)


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory) -> PreTrainedTokenizerFast:
    rng = np.random.default_rng(0)
    corpus = [toy_text(rng, bool(index % 2)) for index in range(200)]
    model = Tokenizer(models.BPE(unk_token="<unk>"))
    model.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    model.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=["<s>", "<pad>", "</s>", "<unk>"]))
    model.post_processor = processors.TemplateProcessing(single="<s> $A </s>", special_tokens=[("<s>", 0), ("</s>", 2)])
    wrapped = PreTrainedTokenizerFast(tokenizer_object=model, bos_token="<s>", eos_token="</s>", pad_token="<pad>", unk_token="<unk>")
    path = tmp_path_factory.mktemp("tokenizer")
    wrapped.save_pretrained(path)
    wrapped.saved_path = path.as_posix()
    return wrapped


@pytest.fixture(scope="module")
def plain_tokenizer(tokenizer, tmp_path_factory):
    """The same vocabulary without special tokens, as GPT-2 tokenizes."""
    raw = Tokenizer.from_str(tokenizer.backend_tokenizer.to_str())
    raw.post_processor = None
    return PreTrainedTokenizerFast(tokenizer_object=raw, pad_token="<pad>", eos_token="</s>")


def tiny_lm(seed: int) -> GPT2LMHeadModel:
    torch.manual_seed(seed)
    return GPT2LMHeadModel(GPT2Config(vocab_size=400, n_positions=64, n_embd=32, n_layer=1, n_head=2))


@pytest.fixture(scope="module")
def pair(plain_tokenizer) -> ModelPair:
    return ModelPair(tiny_lm(1), tiny_lm(2), plain_tokenizer, context=16)


def toy_corpus(tmp_path: Path, count: int = 160) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows, texts = [], []
    for index in range(count):
        label = index % 2
        text = toy_text(rng, bool(label))
        path = f"text/{index}"
        split = "train" if index < count * 0.6 else ("val" if index < count * 0.8 else "test")
        rows.append({"path": path, "label": label, "source_dataset": "toy", "generator_name": "gen" if label else "real",
                     "group_id": prompt_group(text), "content_hash": str(index), "domain": "news" if index % 4 < 2 else "essay", "split": split})
        texts.append({"path": path, "text": text})
    (tmp_path / "data" / "toy").mkdir(parents=True)
    pd.DataFrame(texts).to_parquet(tmp_path / "data" / "toy" / "texts.parquet")
    frame = pd.DataFrame(rows)
    frame.to_csv(tmp_path / "toy.csv", index=False)
    return frame


def experiment_config(tmp_path: Path, tokenizer) -> dict:
    values = {
        "tracker": "none", "num_workers": "0", "device": "cpu", "epochs": "10", "patience": "10", "warmup_epochs": "0",
        "model.batch_size": "8", "model.accumulation_steps": "1", "model.learning_rate": "0.001", "max_length": "24", "stride": "8",
        "manifest": (tmp_path / "toy.csv").as_posix(), "data_root": (tmp_path / "data").as_posix(),
        "checkpoint_dir": (tmp_path / "checkpoints").as_posix(), "model.name": "tiny",
    }
    with initialize_config_dir(CONFIG_DIR, version_base=None):
        config = OmegaConf.to_container(compose("train", overrides=[f"{key}={value}" for key, value in values.items()]), resolve=True)
    config["model"].update(overrides=TINY_ENCODER, tokenizer=tokenizer.saved_path)
    return config


def test_split_sentences_keeps_spans():
    text = 'First one. "Second" here! Third? last bit'
    spans = split_sentences(text)
    assert [text[a:b] for a, b in spans] == ["First one.", '"Second" here!', "Third? last bit"]


def test_token_stats_match_a_single_forward_pass(pair):
    text = "The river city was a story."
    ids = pair.tokenizer(text, return_tensors="pt")["input_ids"]
    assert ids.shape[1] < pair.context
    stats = pair.token_stats(text)
    with torch.no_grad():
        expected = torch.log_softmax(pair.performer(ids).logits[0, :-1], -1)
    actual = expected.gather(1, ids[0, 1:, None])[:, 0].numpy()
    np.testing.assert_allclose(stats.log_prob, actual, atol=1e-5)
    np.testing.assert_array_equal(stats.rank == 1, expected.argmax(1).numpy() == ids[0, 1:].numpy())


def test_long_text_is_scored_once_per_token_across_windows(pair):
    text = " ".join(["the river city was a story of music"] * 12)
    count = len(pair.tokenizer(text)["input_ids"])
    assert count > 3 * pair.context
    stats = pair.token_stats(text)
    assert len(stats.log_prob) == count - 1
    assert np.all(np.diff(stats.offsets[:, 0]) > 0)


def test_summary_features(pair):
    stats = pair.token_stats("The river city was a story. A game of music in the market. It was science.")
    features = summarize(stats, stats.sentence_ids(split_sentences("The river city was a story. A game of music in the market. It was science.")))
    assert list(features) == list(FEATURES)
    assert features["rank_1"] <= features["rank_10"] <= features["rank_100"] <= features["rank_1000"] == 1.0
    assert features["binoculars"] == pytest.approx(features["log_ppl"] / stats.cross_entropy.mean())
    assert features["burstiness"] > 0


def feature_frame(rng: np.random.Generator, count: int, domain_shift: float = 0.0) -> pd.DataFrame:
    labels = np.arange(count) % 2
    frame = pd.DataFrame({name: rng.normal(size=count) for name in FEATURES})
    frame["log_ppl"] -= labels * 2.0 + domain_shift
    return frame.assign(label=labels, group_id=[f"g{index}" for index in range(count)],
                        generator_name=np.where(labels, "gen", "real"), domain=np.where(np.arange(count) % 4 < 2, "native", "toefl"))


def test_ensemble_fits_and_reports_false_positives_per_domain(tmp_path):
    rng = np.random.default_rng(0)
    model = ensemble.fit(feature_frame(rng, 200))
    assert ensemble.weights(model)["log_ppl"] < 0  # lower perplexity pushes toward AI
    report = ensemble.evaluate(model, feature_frame(rng, 100), {"test": feature_frame(rng, 100)}, tmp_path)
    assert report["test"]["roc_auc"] > 0.9
    assert {row["domain"] for row in report["test"]["human_false_positive_rate_by_domain"]} == {"native", "toefl"}
    assert (tmp_path / "metrics.json").exists()


def test_sample_keeps_every_generator():
    frame = pd.DataFrame({"label": [0] * 90 + [1] * 10, "generator_name": ["real"] * 90 + ["a"] * 5 + ["b"] * 5, "split": "train"})
    assert set(ensemble.sample(frame, "train", 10, 0)["generator_name"]) == {"real", "a", "b"}


def test_extract_resumes_without_rescoring(tmp_path, pair, monkeypatch):
    frame = toy_corpus(tmp_path, 12)
    calls = []
    monkeypatch.setattr(ensemble, "text_features", lambda p, text: calls.append(text) or text_features(p, text))
    output = tmp_path / "features.parquet"
    ensemble.extract(frame.head(5), tmp_path / "data", output, pair)
    result = ensemble.extract(frame, tmp_path / "data", output, pair)
    assert len(calls) == 12
    assert len(result) == 12 and {"label", "domain", *FEATURES} <= set(result.columns)


def test_cross_dataset_rows_opening_like_training_are_found(tmp_path):
    frame = toy_corpus(tmp_path, 10)
    other = frame[frame["split"] == "test"].assign(group_id="unrelated")
    assert ensemble.opens_like_training(other, tmp_path / "data", frame).sum() == 0
    assert ensemble.opens_like_training(frame[frame["split"] == "train"], tmp_path / "data", frame).all()


def fitted_meta(pair) -> LogisticRegression:
    rng = np.random.default_rng(0)
    model = ensemble.fit(feature_frame(rng, 100))
    return model


def test_analyze_returns_the_full_result(pair):
    text = "The river city was a story. A game of music in the market. It was science. This is the end of it."
    result = detector.TextDetector(pair, fitted_meta(pair), "1.0.0").analyze(text)
    assert 0.0 <= result["confidence_band"]["low"] <= result["confidence_band"]["high"] <= 1.0
    assert result["label"] in {"likely human", "mixed", "likely AI"}
    assert [sentence["text"] for sentence in result["sentences"]] == [text[a:b] for a, b in split_sentences(text)]
    assert result["warnings"] and "academic misconduct" in result["disclaimer"]


def test_label_rules():
    sentence = lambda p: detector.Sentence(0, 1, "one two three four", p)
    assert detector.label_for(0.9, [sentence(0.9)]) == "likely AI"
    assert detector.label_for(0.1, [sentence(0.1)]) == "likely human"
    assert detector.label_for(0.5, [sentence(0.5)]) == "mixed"
    assert detector.label_for(0.7, [sentence(0.9), sentence(0.1)]) == "mixed"


def test_half_precision_checkpoint_loads_as_float32_for_training(tmp_path, tokenizer):
    # deberta-v3-small is published in fp16; GradScaler refuses fp16 weights.
    architecture = AutoConfig.for_model(**TINY_ENCODER, num_labels=1)
    AutoModelForSequenceClassification.from_config(architecture).half().save_pretrained(tmp_path / "fp16")
    config = experiment_config(tmp_path, tokenizer)
    config["model"].update(backbone=(tmp_path / "fp16").as_posix(), overrides=None)
    model, _ = build_model(config)
    assert {parameter.dtype for parameter in model.parameters()} == {torch.float32}


def test_windows_overlap_and_keep_special_tokens(tokenizer):
    windows = encode_windows(tokenizer, " ".join(["river city music"] * 20), 24, 8)
    assert windows.shape[1:] == (2, 24) and len(windows) > 2
    assert (windows[:, 0, 0] == tokenizer.bos_token_id).all()
    first, second = windows[0, 0][windows[0, 1] == 1][1:-1], windows[1, 0][windows[1, 1] == 1][1:-1]
    np.testing.assert_array_equal(first[-8:], second[:8])


@pytest.fixture(scope="module")
def trained(tmp_path_factory, tokenizer):
    tmp_path = tmp_path_factory.mktemp("train")
    toy_corpus(tmp_path)
    config = experiment_config(tmp_path, tokenizer)
    return train(config), config


def test_training_learns_the_marker(trained):
    path, _ = trained
    assert path.name == "text-tiny-1.0.0-best.pt"
    assert torch.load(path, weights_only=False)["best_auc"] > 0.9
    model, tokenizer, _ = load_checkpoint(path)
    rng = np.random.default_rng(5)
    assert score_text(model, tokenizer, toy_text(rng, True), 24, 8) > score_text(model, tokenizer, toy_text(rng, False), 24, 8)


def test_integrated_gradients_are_complete_and_find_the_marker(trained):
    model, tokenizer, _ = load_checkpoint(trained[0])
    result = explain.integrated_gradients(model, tokenizer, "The river " + MARKER + " city was a story.", max_length=24, steps=64)
    assert result["completeness_gap"] < 0.05 * max(1.0, abs(result["logit"]))
    assert result["tokens"][int(np.argmax(result["attributions"]))].lstrip("Ġ") in MARKER


def test_human_edit_swaps_the_requested_share():
    rng = np.random.default_rng(0)
    ai = "A one. A two. A three. A four."
    edited = robustness.human_edit(ai, "H one. H two. H three.", 0.5, rng)
    assert robustness.sentences_of(edited).count("H one.") == 1
    assert sum(sentence.startswith("H") for sentence in robustness.sentences_of(edited)) == 2
    assert robustness.first_words("a b c d", 2) == "a b"


def test_robustness_reports_every_condition(pair):
    rng = np.random.default_rng(0)
    ai = pd.DataFrame({"text": [toy_text(rng, True) for _ in range(6)], "domain": "news"})
    human = pd.DataFrame({"text": [toy_text(rng, False) for _ in range(6)], "domain": "news"})
    report = robustness.run(fitted_meta(pair), lambda text: text_features(pair, text), ai, human, 0.5, paraphrase=str.upper)
    assert {"paraphrased", "human_edited_25", "human_edited_50", "short_50_ai", "short_75_human"} <= set(report)
