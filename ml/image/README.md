# Image detector

Run every command from the repository root. Paths below assume `image_core` has been prepared as described in the [ML guide](../README.md).

## Train on Colab

[notebooks/colab_image.ipynb](../../notebooks/colab_image.ipynb) does the whole run on real images: it copies a seeded subset of GenImage ADM (20,000 fakes and 20,000 ImageNet originals) and 4,000 BigGAN fakes out of the Kaggle archives, prepares them, trains ConvNeXt-Tiny with the frequency branch, then evaluates on the held-out ADM test split and on BigGAN as an unseen generator, exports ONNX and packs the results. The steps below are the same run by hand.

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

## Results

One run on a Kaggle T4: ConvNeXt-Tiny with the frequency branch and `robust` augmentation, trained on 20,000 GenImage ADM images and 20,000 ImageNet originals with a grouped 70/15/15 split. Each epoch of 583 steps took about 202 seconds. Validation AUC peaked at 0.99985 after epoch 4, and training stopped after epoch 9 when it had not improved for five epochs. The threshold was set on validation to reach 95% precision. Full output is in [ml/eval/results/image/](../eval/results/image/).

| Test set | Images | AUC (95% interval) | At the threshold |
|---|---|---|---|
| ADM test split | 3,000 AI, 3,000 real | 0.9985 (0.9972 to 0.9995) | 97.8% accuracy; 99.97% of ADM images and 95.6% of real photos classified correctly |
| BigGAN, never seen in training | 4,000 AI, the same 3,000 real | 0.901 (0.893 to 0.908) | 77.3% of BigGAN images detected |

The drop from ADM to BigGAN is the number to watch: a detector trained on one diffusion model catches about three quarters of a GAN's images.

### Experiment grid

The grid below was then run on the same data and split, on a Kaggle T4. Every run stopped early, five epochs after its best one, before the 75-minute budget. Each configuration was trained once, so treat differences of a few points in BigGAN AUC as possible noise.

| Backbone | DCT branch | Augmentation | ADM test AUC | BigGAN AUC (unseen) | BigGAN detected |
|---|---|---|---|---|---|
| EfficientNet-B0 | no | robust | 0.9991 | 0.935 | 75.3% |
| EfficientNet-B0 | yes | robust | 0.9995 | 0.918 | 71.6% |
| ConvNeXt-Tiny | no | robust | 0.9992 | 0.956 | 86.0% |
| ConvNeXt-Tiny | yes | robust | 0.9985 | 0.901 | 77.3% |
| ConvNeXt-Tiny | yes | none | 1.0000 | 0.650 | 12.5% |
| ConvNeXt-Tiny | yes | robust without JPEG | 0.9994 | 0.958 | 87.0% |
| ViT-B/16 | no | robust | 0.9986 | 0.959 | 80.8% |
| ViT-B/16 | yes | robust | 0.9979 | **0.980** | **89.9%** |

- ADM can't rank the models: all of them score 0.998 or higher.
- Augmentation matters most. Without it the model still separates ADM perfectly but catches 12.5% of BigGAN images.
- The DCT branch helps ViT-B/16 and hurts the two CNNs on BigGAN.
- Removing JPEG recompression raised BigGAN AUC from 0.901 to 0.958. That contradicts the reason for the augmentation, so it needs more seeds before it is trusted.

The full table, with precision, recall and calibration error, is [ml/eval/results/image_models.csv](../eval/results/image_models.csv). CPU latency was not measured for these runs.

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
