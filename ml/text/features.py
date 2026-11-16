"""Zero-shot statistics from two small causal language models.

Every token is scored once, with up to `context` tokens of left context, by a
performer (GPT-2) and an observer (DistilGPT-2). They share a tokenizer, so
their next-token distributions line up position by position. From those
per-token numbers come:

- log perplexity and its spread across sentences (burstiness)
- GLTR rank buckets: the share of tokens that were the model's top 1, 10, 100
  or 1000 guess
- mean predictive entropy
- a Binoculars score: performer log perplexity divided by the observer-to-
  performer cross-entropy (Hans et al., 2024). Low values point to machine text.

Per-token values are kept, so a sentence or a resample of sentences can be
scored again without running either model.
"""

import re
from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

FEATURES = ("log_ppl", "burstiness", "rank_1", "rank_10", "rank_100", "rank_1000", "entropy", "binoculars")
RANK_BUCKETS = (1, 10, 100, 1000)
PERFORMER = "gpt2"
OBSERVER = "distilgpt2"
CONTEXT = 512
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Character spans of sentences. A rule, not a parser: abbreviations such as "Dr. Smith" split early."""
    spans, start = [], 0
    for match in _SENTENCE_END.finditer(text):
        spans.append((start, match.start()))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(begin, end) for begin, end in spans if text[begin:end].strip()]


@dataclass
class TokenStats:
    """One row per scored token. The first token has no left context and is not scored."""

    offsets: np.ndarray  # [n, 2] character span of each scored token
    log_prob: np.ndarray  # performer log probability of the actual token
    rank: np.ndarray  # performer rank of the actual token, 1 = top guess
    entropy: np.ndarray  # performer predictive entropy
    cross_entropy: np.ndarray  # observer probabilities against performer log probabilities

    def subset(self, mask: np.ndarray) -> "TokenStats":
        return TokenStats(self.offsets[mask], self.log_prob[mask], self.rank[mask], self.entropy[mask], self.cross_entropy[mask])

    def sentence_ids(self, spans: list[tuple[int, int]]) -> np.ndarray:
        """Sentence index of each token, by where the token starts; -1 between sentences."""
        ids = np.full(len(self.offsets), -1)
        for index, (begin, end) in enumerate(spans):
            ids[(self.offsets[:, 0] >= begin) & (self.offsets[:, 0] < end)] = index
        return ids


class ModelPair:
    """Performer and observer with their shared tokenizer, in eval mode on CPU unless told otherwise."""

    def __init__(self, performer, observer, tokenizer: PreTrainedTokenizerBase, context: int = CONTEXT, device: str = "cpu"):
        self.performer = performer.eval().to(device)
        self.observer = observer.eval().to(device)
        self.tokenizer = tokenizer
        self.context = context
        self.device = device

    @classmethod
    def pretrained(cls, performer: str = PERFORMER, observer: str = OBSERVER, device: str = "cpu") -> "ModelPair":
        tokenizer = AutoTokenizer.from_pretrained(performer)
        return cls(AutoModelForCausalLM.from_pretrained(performer), AutoModelForCausalLM.from_pretrained(observer), tokenizer, device=device)

    @torch.inference_mode()
    def token_stats(self, text: str) -> TokenStats:
        encoded = self.tokenizer(text, return_offsets_mapping=True)
        ids = torch.tensor(encoded["input_ids"])
        offsets = np.array(encoded["offset_mapping"]).reshape(-1, 2)
        rows: list[np.ndarray] = []
        # Window i predicts tokens [scored_from, end); later windows keep half a window of context.
        stride = self.context // 2
        scored_from = 1
        while scored_from < len(ids):
            begin = max(0, scored_from - stride) if scored_from > 1 else 0
            end = min(len(ids), begin + self.context)
            window = ids[begin:end].unsqueeze(0).to(self.device)
            performer = torch.log_softmax(self.performer(window).logits[0].float(), -1)
            observer = torch.softmax(self.observer(window).logits[0].float(), -1)
            # Logit at position p predicts token p + 1.
            first = scored_from - begin - 1
            predicted, targets = performer[first:-1], window[0, first + 1 :]
            actual = predicted.gather(1, targets[:, None])[:, 0]
            columns = [
                actual,
                (predicted > actual[:, None]).sum(1) + 1.0,
                -(predicted.exp() * predicted).sum(1),
                -(observer[first:-1] * predicted).sum(1),
            ]
            rows.append(torch.stack(columns, 1).cpu().numpy())
            scored_from = end
        table = np.concatenate(rows) if rows else np.zeros((0, 4))
        return TokenStats(offsets[1 : 1 + len(table)], table[:, 0], table[:, 1].astype(int), table[:, 2], table[:, 3])


def summarize(stats: TokenStats, sentence_ids: np.ndarray | None = None) -> dict[str, float]:
    """Text-level features from per-token values."""
    if len(stats.log_prob) == 0:
        return dict.fromkeys(FEATURES, 0.0)
    log_ppl = float(-stats.log_prob.mean())
    features = {"log_ppl": log_ppl, "entropy": float(stats.entropy.mean())}
    for bucket in RANK_BUCKETS:
        features[f"rank_{bucket}"] = float((stats.rank <= bucket).mean())
    features["binoculars"] = log_ppl / max(float(stats.cross_entropy.mean()), 1e-6)
    per_sentence = []
    if sentence_ids is not None:
        per_sentence = [-stats.log_prob[sentence_ids == index].mean() for index in np.unique(sentence_ids[sentence_ids >= 0])]
    features["burstiness"] = float(np.std(per_sentence)) if len(per_sentence) > 1 else 0.0
    return {name: features[name] for name in FEATURES}


def text_features(pair: ModelPair, text: str) -> dict[str, float]:
    stats = pair.token_stats(text)
    return summarize(stats, stats.sentence_ids(split_sentences(text)))
