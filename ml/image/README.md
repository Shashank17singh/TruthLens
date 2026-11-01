# Image detector

Run every command from the repository root. Paths below assume `image_core` has been prepared as described in the [ML guide](../README.md).

## Train on Colab

Save checkpoints to Google Drive, because a free session's disk disappears when it ends.

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
pip install -r ml/requirements.txt
wandb login
python -m ml.image.train model=convnext_tiny use_frequency=true \
  checkpoint_dir=/content/drive/MyDrive/truthlens/checkpoints steps_per_epoch=1500
```

Running the same command again resumes from the last finished epoch. `steps_per_epoch` caps an epoch so a run fits inside one session.

## The experiment grid

Architectures, with and without the frequency branch:

```bash
python -m ml.image.train -m model=efficientnet_b0,convnext_tiny,vit_b16 use_frequency=true,false
```

Augmentation ablation, on the best architecture:

```bash
python -m ml.image.train model=convnext_tiny augment=none
python -m ml.image.train model=convnext_tiny augment.jpeg.enabled=false augment.name=no_jpeg
```

Always set `augment.name` when you change a single flag. The run name comes from it, so without a new name the command would resume the full `robust` run.

## Evaluate and export

```bash
python -m ml.image.evaluate checkpoints/image/<run>/<run>-best.pt
python -m ml.image.export checkpoints/image/<run>/<run>-best.pt
python -m ml.image.results_table
```

| Step | Output |
|---|---|
| `evaluate` | `ml/eval/results/image/<run>/`: `metrics.json` and five figures |
| `export` | `checkpoints/onnx/<run>.onnx` and a `.json` file with the checksum, parity result and latency |
| `results_table` | `ml/eval/results/image_models.csv` and `paper/tables/image_models.tex` |

Export stops with an error if ONNX and PyTorch differ by more than 1e-3.

## Serving

The ONNX model takes a float32 tensor shaped `[batch, 3, 224, 224]` and returns one logit per image. Before inference, scale pixels to 0–1, then apply the `mean` and `std` from the export `.json`. They differ by backbone.

After inference, apply `calibration` and `threshold_choice.threshold` from `metrics.json`:

```python
probability = sigmoid(logit / temperature + bias)
is_fake = probability >= threshold
```
