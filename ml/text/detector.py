"""Score one text: overall probability, label, confidence band and per-sentence scores.

Both language models run once over the whole text. Sentences and the
confidence band reuse those per-token numbers, so they cost no extra passes:

- a sentence's score is the meta-classifier on that sentence's tokens, which
  were still predicted with the full left context;
- the band is a bootstrap over sentences. Short texts have few sentences, so
  their bands come out wide, which is the honest answer.
"""

from dataclasses import asdict, dataclass

import numpy as np

from ml.text.ensemble import TRANSFORMER
from ml.text.features import FEATURES, ModelPair, TokenStats, split_sentences, summarize

DISCLAIMER = (
    "This result is probabilistic. Detectors make mistakes, and they flag non-native English writers "
    "more often. Never use it alone to decide academic misconduct or any other penalty."
)
HUMAN_BELOW = 0.35
AI_FROM = 0.65
MIXED_SHARE = 0.25  # each side of a split verdict must cover this share of the words
BOOTSTRAP_ROUNDS = 200
SHORT_WORDS = 100


@dataclass
class Sentence:
    start: int
    end: int
    text: str
    probability: float


def label_for(probability: float, sentences: list[Sentence]) -> str:
    """`mixed` for a middling score, or when a quarter of the words reads as AI and a quarter as human."""
    words = np.array([len(sentence.text.split()) for sentence in sentences], dtype=float)
    scores = np.array([sentence.probability for sentence in sentences])
    if len(sentences) > 1 and words.sum() > 0:
        ai, human = words[scores >= AI_FROM].sum() / words.sum(), words[scores < HUMAN_BELOW].sum() / words.sum()
        if ai >= MIXED_SHARE and human >= MIXED_SHARE:
            return "mixed"
    if probability >= AI_FROM:
        return "likely AI"
    if probability < HUMAN_BELOW:
        return "likely human"
    return "mixed"


class TextDetector:
    def __init__(self, pair: ModelPair, meta, version: str, seed: int = 1337):
        if TRANSFORMER in meta.feature_names:
            raise ValueError("this meta-classifier needs a transformer logit; use one trained on statistics only")
        self.pair = pair
        self.meta = meta
        self.version = version
        self.seed = seed

    def probability(self, stats: TokenStats, sentence_ids: np.ndarray) -> float:
        features = summarize(stats, sentence_ids)
        return float(self.meta.predict_proba(np.array([[features[name] for name in FEATURES]]))[0, 1])

    def analyze(self, text: str) -> dict:
        stats = self.pair.token_stats(text)
        spans = split_sentences(text)
        ids = stats.sentence_ids(spans)
        overall = self.probability(stats, ids)

        sentences = []
        for index, (start, end) in enumerate(spans):
            mask = ids == index
            if mask.any():
                sentences.append(Sentence(start, end, text[start:end], self.probability(stats.subset(mask), ids[mask])))

        rng = np.random.default_rng(self.seed)
        present = np.unique(ids[ids >= 0])
        resampled = []
        for _ in range(BOOTSTRAP_ROUNDS if len(present) > 1 else 0):
            chosen = rng.choice(present, len(present))
            # Each draw keeps its own sentence id, so a sentence drawn twice counts twice in burstiness.
            masks = [np.flatnonzero(ids == sentence) for sentence in chosen]
            order = np.concatenate(masks)
            relabelled = np.concatenate([np.full(len(mask), draw) for draw, mask in enumerate(masks)])
            resampled.append(self.probability(stats.subset(order), relabelled))
        low, high = (np.percentile(resampled, [5, 95]) if resampled else (0.0, 1.0))

        words = len(text.split())
        return {
            "probability": overall,
            "label": label_for(overall, sentences),
            "confidence_band": {"low": float(low), "high": float(high), "level": 0.9},
            "sentences": [asdict(sentence) for sentence in sentences],
            "words": words,
            "warnings": [f"Under {SHORT_WORDS} words: too short for a reliable score."] if words < SHORT_WORDS else [],
            "disclaimer": DISCLAIMER,
            "model_version": self.version,
        }
