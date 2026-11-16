"""Robustness of the meta-classifier at its validation threshold.

    python -m ml.text.robustness --model checkpoints/text/meta.joblib --features features/mage-test.parquet \
        --threshold-from ml/eval/results/text/metrics.json --limit 200 --output ml/eval/results/text/robustness.json

Conditions, each scored on the same AI and human test texts:
- paraphrased: every AI sentence rewritten by a T5 paraphraser
- human_edited_25 / _50: that share of AI sentences swapped for sentences of a
  human text from the same domain, the way a writer mixes in their own work
- short_50 / short_75: only the first 50 or 75 words, for AI and human texts
The non-native English check runs in `ensemble evaluate` on detector_bias and
is reported per domain there.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ml.common.training import resolve_device
from ml.text.data import read_texts
from ml.text.ensemble import probabilities
from ml.text.features import FEATURES, ModelPair, split_sentences, text_features

PARAPHRASER = "humarin/chatgpt_paraphraser_on_T5_base"
SHORT_LENGTHS = (50, 75)
EDIT_SHARES = (0.25, 0.5)


def sentences_of(text: str) -> list[str]:
    return [text[start:end].strip() for start, end in split_sentences(text)]


def first_words(text: str, count: int) -> str:
    return " ".join(text.split()[:count])


def human_edit(ai_text: str, human_text: str, share: float, rng: np.random.Generator) -> str:
    """Replace `share` of the AI sentences, at random positions, with human sentences in order."""
    ai, human = sentences_of(ai_text), sentences_of(human_text)
    count = min(len(human), max(1, round(len(ai) * share)))
    for position, sentence in zip(sorted(rng.choice(len(ai), count, replace=False)), human):
        ai[position] = sentence
    return " ".join(ai)


def t5_paraphraser(name: str = PARAPHRASER, device: str = "cpu") -> Callable[[str], str]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSeq2SeqLM.from_pretrained(name).eval().to(device)

    def paraphrase(text: str) -> str:
        sentences = sentences_of(text)
        batch = tokenizer([f"paraphrase: {sentence}" for sentence in sentences], return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
        output = model.generate(**batch, max_length=128, num_beams=4, repetition_penalty=1.5)
        return " ".join(tokenizer.batch_decode(output, skip_special_tokens=True))

    return paraphrase


def conditions(ai: pd.DataFrame, human: pd.DataFrame, paraphrase: Callable[[str], str] | None, seed: int) -> dict[str, tuple[list[str], int]]:
    """Condition name to (texts, label). `ai` and `human` carry `text` and `domain`."""
    rng = np.random.default_rng(seed)
    by_domain = {domain: group["text"].tolist() for domain, group in human.groupby("domain")}
    pool = human["text"].tolist()

    def partner(domain: str) -> str:
        candidates = by_domain.get(domain) or pool
        return candidates[int(rng.integers(len(candidates)))]

    result = {}
    if paraphrase is not None:
        result["paraphrased"] = ([paraphrase(text) for text in ai["text"]], 1)
    for share in EDIT_SHARES:
        result[f"human_edited_{round(share * 100)}"] = ([human_edit(text, partner(domain), share, rng) for text, domain in zip(ai["text"], ai["domain"])], 1)
    for length in SHORT_LENGTHS:
        result[f"short_{length}_ai"] = ([first_words(text, length) for text in ai["text"]], 1)
        result[f"short_{length}_human"] = ([first_words(text, length) for text in human["text"]], 0)
    return result


def run(meta, score_features: Callable[[str], dict], ai: pd.DataFrame, human: pd.DataFrame, threshold: float, paraphrase=None, seed: int = 1337) -> dict:
    """Share of each condition's texts given their true label at the threshold."""
    def flagged(texts: list[str]) -> np.ndarray:
        frame = pd.DataFrame([score_features(text) for text in texts])
        return probabilities(meta, frame) >= threshold

    report = {"threshold": threshold, "baseline_ai_detected": float(flagged(ai["text"].tolist()).mean()),
              "baseline_human_flagged": float(flagged(human["text"].tolist()).mean())}
    for name, (texts, label) in conditions(ai, human, paraphrase, seed).items():
        rate = flagged(texts).mean()
        report[name] = {"samples": len(texts), ("ai_detected" if label else "human_flagged"): float(rate)}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True, help="extracted test features; their texts are perturbed")
    parser.add_argument("--threshold-from", type=Path, required=True, help="metrics.json from ensemble evaluate")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--limit", type=int, default=200, help="texts per label")
    parser.add_argument("--no-paraphrase", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", help="auto picks CUDA when it is available")
    args = parser.parse_args()

    meta = joblib.load(args.model)
    assert list(meta.feature_names) == list(FEATURES), "robustness runs on the statistics-only ensemble"
    frame = pd.read_parquet(args.features)
    frame = frame.assign(text=read_texts(frame, args.data_root))
    ai = frame[frame["label"] == 1].sample(min(args.limit, (frame["label"] == 1).sum()), random_state=args.seed)
    human = frame[frame["label"] == 0].sample(min(args.limit, (frame["label"] == 0).sum()), random_state=args.seed)
    device = resolve_device(args.device)
    pair = ModelPair.pretrained(device=device)
    threshold = json.loads(args.threshold_from.read_text(encoding="utf-8"))["threshold"]
    paraphrase = None if args.no_paraphrase else t5_paraphraser(device=device)
    report = run(meta, lambda text: text_features(pair, text), ai, human, threshold, paraphrase, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
