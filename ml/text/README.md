# Text detector

Run every command from the repository root, after preparing `mage`, `hc3` and `detector_bias` as described in the [ML guide](../README.md).

The detector has three parts:

1. **Statistics.** GPT-2 and DistilGPT-2 give eight features per text: log perplexity, burstiness, four GLTR rank buckets, mean entropy and a Binoculars score. They run on a CPU.
2. **Transformers.** RoBERTa-base or DeBERTa-v3-small, fine-tuned on 512-token windows. These need a GPU, so they train on Colab or Kaggle.
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

## Results

The models were trained on Kaggle with a T4 GPU using [notebooks/colab_text.ipynb](../../notebooks/colab_text.ipynb). RoBERTa-base was fine-tuned for 4 epochs of 3,000 steps on MAGE, and its best validation AUC was 0.996, at epoch 2. DeBERTa-v3-small, trained the same way in a second run, reached 0.9956 at epoch 3, just under RoBERTa's 0.9962, so the ensemble uses RoBERTa.

**Ensemble with RoBERTa.** This is a logistic regression over the eight statistics plus the RoBERTa logit, trained on 4,999 MAGE texts. The threshold of 0.732 is the validation equal-error point. The file is `ml/eval/results/text/metrics.json`.

| Test set | Texts | AUC (95% interval) | At the threshold |
|---|---|---|---|
| MAGE validation | 2,001 | 0.996 (0.994 to 0.997) | accuracy 0.969 |
| MAGE test | 3,999 | 0.997 (0.997 to 0.998) | accuracy 0.976 |
| MAGE, unseen LLaMA generators | 1,999 AI | none, AI texts only | 90.1% detected |
| HC3, excluding rows that open like MAGE training texts | 2,000 | 0.997 (0.995 to 0.998) | 99.3% of AI detected, 9.3% of Reddit ELI5 human answers flagged |
| ChatGPT Detector Bias | 831 | 0.794 (0.761 to 0.825) | accuracy 0.752 |

Human texts flagged as AI on the bias set:

| Writers | Essays | Flagged as AI |
|---|---|---|
| TOEFL essays, non-native writers | 89 | 61.8% |
| College essays | 70 | 14.3% |
| Stanford CS224N student writing | 145 | 8.3% |
| US 8th-grade essays (Hewlett) | 88 | 5.7% |

The transformer cuts false positives for native writers. It still flags most TOEFL essays, so the bias that Liang et al. reported for commercial detectors remains. The detector must not be used to judge non-native writers, and the disclaimer says so.

**Statistics only.** This is the same regression without the transformer, trained on 4,999 MAGE texts. It scores 0.785 AUC on validation (2,001 texts) and 0.782 (0.767 to 0.797) on the test set (3,999 texts), with accuracy 0.721 at its threshold of 0.617. The file is `ml/eval/results/text/statistics/metrics.json`.

Robustness of the statistics-only model, on 300 AI and 300 human MAGE test texts (`statistics/robustness.json`):

| Condition | Result |
|---|---|
| Full AI texts | 71.3% detected |
| Paraphrased by T5 | 47.7% detected |
| A quarter of the sentences replaced by human writing | 47.0% detected |
| Half of the sentences replaced | 37.3% detected |
| First 75 words | 53.7% of AI detected, 23.3% of human flagged |
| First 50 words | 37.3% of AI detected, 22.7% of human flagged |
| Full human texts | 30.7% flagged |

The robustness test has not been run on the ensemble with RoBERTa. DeBERTa-v3-small's first run crashed: it is published in fp16, transformers 5 kept that precision, and mixed-precision training refused to unscale its gradients. The loader now forces float32, which fixed the second run. On the development laptop, one statistics-only `analyze` call took 0.8 seconds for a 79-word text and 10.7 seconds for a 638-word text, while the laptop was short of memory.

## Robustness

```bash
python -m ml.text.robustness --model checkpoints/text/meta.joblib --features data/features/text/mage-test.parquet \
  --threshold-from ml/eval/results/text/statistics/metrics.json --limit 300 --output ml/eval/results/text/statistics/robustness.json
```

It reports detection rates for AI text that has been paraphrased by a T5 model, AI text with a quarter or half of its sentences swapped for human sentences from the same domain, and AI and human texts cut to their first 50 or 75 words. Add `--no-paraphrase` to skip the 900 MB paraphraser download.

## Transformers on a GPU

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
