# TruthLens ML

Run every command from the repository root.

## Setup

Local machine:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r ml/requirements.txt
```

Colab or Kaggle already has PyTorch:

```bash
pip install -r ml/requirements.txt
```

## Prepare a dataset

```bash
python -m ml.common.prepare mage --workers 4
python -m ml.common.prepare mage --limit 5000    # Stratified subset for a quick check
```

Each run does the following, in order:

1. Downloads the files, or finds the ones you placed by hand.
2. Verifies checksums. A file with no published checksum is recorded in `ml/common/checksums.lock.json`, so commit that file after the first run.
3. Reads the raw layout through the dataset's adapter.
4. Preprocesses each sample and writes it to `data/processed/<dataset>/`.
5. Assigns splits and refuses to write a manifest that leaks.
6. Writes `data/manifests/<dataset>.csv`, a balance report and `docs/datacards/<dataset>.md`.

`data/` is ignored by git. Paths in a manifest are relative to `data/processed/<dataset>/`.

Datasets marked `manual` in `ml/common/datasets.yaml` stop with instructions until their files are in `data/raw/<dataset>/`.

## Manifest columns

| Column | Meaning |
|---|---|
| `path` | Processed file, or `text/<hash>` as a key into `texts.parquet` |
| `label` | `0` real, `1` fake, in every dataset |
| `source_dataset` | Dataset id |
| `generator_name` | Attack or model name, `real`, or `unknown` |
| `group_id` | Identity, speaker or prompt that must stay in one split |
| `domain` | Text domain, audio codec, or `face` / `whole_image` |
| `content_hash` | SHA-256 of the prepared content |
| `split` | `train`, `val`, `test` or `unseen_test` |

## Use a manifest

```python
from pathlib import Path

from ml.common.balance import weighted_sampler
from ml.common.manifest import read_manifest, unseen_evaluation_frame

frame = read_manifest(Path("data/manifests/mage.csv"))
train = frame[frame["split"] == "train"].reset_index(drop=True)
sampler = weighted_sampler(train, seed=1337)
unseen = unseen_evaluation_frame(frame)
```

`unseen_test` holds fakes only. `unseen_evaluation_frame` adds the real test samples, because AUC needs both classes.

## Combine datasets

```bash
python -m ml.common.prepare --combine image_core faceforensics genimage
```

The merged manifest goes through the leakage check again, so content shared between datasets is caught.

## Training

- [Image detector](image/README.md)

## Tests

```bash
python -m pytest
python -m pytest -m "not network"    # Skips the RetinaFace weight download
```
