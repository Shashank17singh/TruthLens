# Text detector

Run every command from the repository root, after preparing `mage`, `hc3` and `detector_bias` as described in the [ML guide](../README.md).

The detector has three parts:

1. **Statistics.** GPT-2 and DistilGPT-2 give eight features per text: log perplexity, burstiness, four GLTR rank buckets, mean entropy and a Binoculars score. They run on a CPU.
2. **Transformers.** RoBERTa-base or DeBERTa-v3-small, fine-tuned on 512-token windows. These need a GPU, so they train on Colab.
3. **Ensemble.** A logistic regression over the statistics, with the transformer logit as one more column when a fine-tuned model exists.

## Statistics and the ensemble on a CPU

```bash
python -m ml.text.ensemble extract --manifest data/manifests/mage.csv --split train --limit 1500 --output data/features/text/mage-train.parquet
python -m ml.text.ensemble extract --manifest data/manifests/mage.csv --split val --limit 500 --output data/features/text/mage-val.parquet
python -m ml.text.ensemble extract --manifest data/manifests/mage.csv --split test --limit 800 --output data/features/text/mage-test.parquet
python -m ml.text.ensemble extract --manifest data/manifests/hc3.csv --limit 500 --exclude-groups-of data/manifests/mage.csv --output data/features/text/hc3.parquet
python -m ml.text.ensemble extract --manifest data/manifests/detector_bias.csv --output data/features/text/detector_bias.parquet

python -m ml.text.ensemble fit --train data/features/text/mage-train.parquet --output checkpoints/text/meta.joblib
python -m ml.text.ensemble evaluate --model checkpoints/text/meta.joblib --validation data/features/text/mage-val.parquet \
  --test data/features/text/mage-test.parquet data/features/text/hc3.parquet data/features/text/detector_bias.parquet \
  --output ml/eval/results/text
```

`--limit` draws a sample stratified by label and generator. Extraction saves every 100 texts and resumes where it stopped. `--exclude-groups-of` drops the HC3 rows that open with the same words as MAGE training texts.

`metrics.json` holds the threshold, which is set at the validation equal-error point, plus AUC with a grouped bootstrap interval, per-generator accuracy, and the share of human texts flagged as AI in each domain. On `detector_bias` those domains separate TOEFL essays by non-native writers from native essays, which is the bias check.

## Results of the CPU run

Statistics-only ensemble, without a transformer. It was trained on 1,501 MAGE texts and scored on samples stratified by generator. The threshold of 0.617 is the validation equal-error point. The files are `ml/eval/results/text/metrics.json` and `robustness.json`.

| Test set | Texts | AUC (95% interval) | At the threshold |
|---|---|---|---|
| MAGE validation | 503 | 0.771 (0.726 to 0.810) | accuracy 0.718 |
| MAGE test | 800 | 0.768 (0.737 to 0.801) | accuracy 0.704 |
| MAGE, unseen LLaMA generators | 500 AI | none, AI texts only | 81.4% detected |
| HC3, excluding the 497 overlapping rows | 500 | 0.932 (0.910 to 0.952) | 95.8% of AI detected, 37% of Reddit ELI5 human answers flagged |
| ChatGPT Detector Bias | 831 | 0.541 (0.498 to 0.581) | close to chance |

Human texts flagged as AI on the bias set:

| Writers | Essays | Flagged as AI |
|---|---|---|
| TOEFL essays, non-native writers | 89 | 86.5% |
| US 8th-grade essays (Hewlett) | 88 | 25.0% |
| College essays | 70 | 25.7% |
| Stanford CS224N student writing | 145 | 0.0% |

Statistical detectors penalize plain, predictable vocabulary, and this reproduces the bias that Liang et al. reported for commercial detectors. The detector must not be used to judge non-native writers, and the disclaimer says so.

Robustness on 150 AI and 150 human MAGE test texts:

| Condition | Result |
|---|---|
| Full AI texts | 73.3% detected |
| A quarter of the sentences replaced by human writing | 49.3% detected |
| Half of the sentences replaced | 36.0% detected |
| First 75 words | 50.7% of AI detected, 28.0% of human flagged |
| First 50 words | 36.7% of AI detected, 22.7% of human flagged |
| Full human texts | 32.0% flagged |

The T5 paraphrase test was not run. The paraphraser and both GPT-2 models do not fit in memory together on the 8 GB development laptop, so run it on Colab. One `analyze` call took 0.8 seconds for a 79-word text and 10.7 seconds for a 638-word text on that laptop while it was short of memory.

## Robustness

```bash
python -m ml.text.robustness --model checkpoints/text/meta.joblib --features data/features/text/mage-test.parquet \
  --threshold-from ml/eval/results/text/metrics.json --limit 200 --output ml/eval/results/text/robustness.json
```

It reports detection rates for AI text that has been paraphrased by a T5 model, AI text with a quarter or half of its sentences swapped for human sentences from the same domain, and AI and human texts cut to their first 50 or 75 words. Add `--no-paraphrase` to skip the 900 MB paraphraser download.

## Transformers on Colab

[notebooks/colab_text.ipynb](../../notebooks/colab_text.ipynb) prepares the data, trains both models with checkpoints on Drive, and rebuilds the ensemble with the transformer logit.

```bash
python -m ml.text.train model=roberta steps_per_epoch=3000
python -m ml.text.train model=deberta
```

## One text

```python
import joblib
from ml.text.detector import TextDetector
from ml.text.features import ModelPair

detector = TextDetector(ModelPair.pretrained(), joblib.load("checkpoints/text/meta.joblib"), "1.0.0")
result = detector.analyze(text)
```

The result holds `probability`, `label` (likely human, mixed or likely AI), a 90% `confidence_band`, per-sentence scores with character offsets, warnings for texts under 100 words, and a `disclaimer`. Scores are probabilistic and must never be used alone to decide academic misconduct.

`ml.text.explain.integrated_gradients` attributes a fine-tuned transformer's logit to tokens.
