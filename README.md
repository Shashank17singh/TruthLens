# TruthLens

Detects AI-generated and manipulated images, audio and text, and explains each result instead of only giving a score.

## Overview

Detectors that score well in a paper often fail on a new generator, or after an image has been compressed by a social network. TruthLens is built around measuring that gap. Every dataset goes through a pipeline that refuses to write a split if the same identity, speaker or prompt reaches both training and evaluation, and every experiment holds out generators the model never sees.

Training runs on free Colab or Kaggle GPUs, and inference is designed to run on CPU through ONNX Runtime.

> TruthLens is in development. The data pipeline and image detector are built and tested, but the image models have not been trained on real data yet, so there are no accuracy results to report. See [Roadmap](#roadmap) and [Known Limitations](#known-limitations).

---

## Architecture Overview

```mermaid
graph TD
    subgraph "Data Pipeline"
    A[Public Datasets] -->|Checksum Verified| B(Dataset Adapters)
    B --> C{Preprocess}
    C --> D{Leak-Free Splits}
    D --> E[Manifests and Data Cards]
    end

    subgraph "Image Detector"
    E --> F(RGB Backbone)
    E --> G(DCT Frequency Branch)
    F --> H{Late Fusion}
    G --> H
    H --> I[Calibration and Threshold]
    I --> J[ONNX Export]
    end

    subgraph "Planned"
    J --> K[Audio and Text Models]
    K --> L[FastAPI and Celery Service]
    L --> M[Web App and Chrome Extension]
    end

    classDef io fill:#f9f0ff,stroke:#8a2be2,stroke-width:2px,color:#000;
    classDef core fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000;
    classDef logic fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000;

    class A,E,J io;
    class B,F,G,I core;
    class C,D,H logic;
```

- The frequency branch computes a DCT inside the model, so training and serving run the same code.
- Temperature, bias and the decision threshold are chosen on validation data only, so the test set cannot influence them.

---

## Features

|                                |                                                                                                           |
| ------------------------------ | --------------------------------------------------------------------------------------------------------- |
| **Leak-Free Splits**           | Whole identities, speakers and prompts stay in one split, and a manifest that leaks is refused            |
| **Unseen-Generator Testing**   | Held-out generators never reach training, and neither do the identities in their samples                  |
| **Dataset Adapters**           | Ten datasets across image, audio and text, with verified checksums and resumable downloads                |
| **Balanced Sampling**          | Each label gets half the draws, shared equally across its generators                                      |
| **Data Cards**                 | A generated datasheet per dataset covering size, classes, licence and known biases                        |
| **Three Architectures**        | EfficientNet-B0, ConvNeXt-Tiny and ViT-B/16, each with or without the frequency branch                    |
| **Robustness Augmentations**   | JPEG compression, blur, resize, crop and colour jitter, each switchable for an ablation study             |
| **Colab-Friendly Training**    | Mixed precision, gradient accumulation, early stopping, and resuming after a session ends                 |
| **Calibrated Probabilities**   | A fitted temperature and bias, with reliability diagrams before and after                                 |
| **Honest Evaluation**          | Per-generator accuracy, AUC intervals that resample whole identities, and warnings on chance-level models |
| **Verified ONNX Export**       | Each export must match PyTorch within 1e-3, and it records its checksum and CPU latency                   |
| **Generated Results Tables**   | CSV and LaTeX tables built from evaluation output, so numbers are never typed by hand                     |

---

## Measured Results

These are real measurements. Accuracy will be added once the models are trained on real data.

### CPU Inference

ONNX Runtime with ImageNet weights at 224×224, on one thread of an Intel Core i5-10210U laptop. All six variants matched PyTorch within 2.3e-5.

| Model           | Size (MB) | RGB (ms) | RGB + Frequency (ms) |
| --------------- | --------- | -------- | -------------------- |
| EfficientNet-B0 | 18        | 27       | 45                   |
| ConvNeXt-Tiny   | 114       | 179      | 177                  |
| ViT-B/16        | 346       | 480      | 553                  |

Times are medians, and size includes the frequency branch.

### Prepared Text Data

| Dataset                 | Input   | Prepared | Use                                        |
| ----------------------- | ------- | -------- | ------------------------------------------ |
| MAGE                    | 432,682 | 301,148  | Training, with the LLaMA family held out   |
| HC3                     | 85,449  | 60,400   | Cross-dataset test                         |
| ChatGPT Detector Bias   | 837     | 831      | False-positive test on non-native writers  |

The leakage check rejected MAGE's published split, because 13,144 rows (3.96%) shared a prompt across splits. TruthLens uses its own split by prompt, so its MAGE scores will not be comparable with the MAGE paper.

---

## Tech Stack

| Component       | Technology                                                |
| --------------- | --------------------------------------------------------- |
| Deep Learning   | Python 3.11, PyTorch, timm                                |
| Data            | pandas, Albumentations, librosa, soundfile, OpenCV        |
| Face Detection  | RetinaFace through batch-face                             |
| Text Processing | langdetect, PyArrow                                       |
| Experiments     | Hydra, Weights & Biases                                   |
| Evaluation      | scikit-learn, SciPy, Matplotlib                           |
| Inference       | ONNX, ONNX Runtime                                        |
| Testing         | Pytest                                                    |

---

## Project Structure

```
TruthLens/
├── ml/
│   ├── common/           # Manifests, splits, checksums, sampling, metrics, data cards
│   ├── image/            # Detector, training, evaluation, ONNX export, results tables
│   │   └── configs/      # Hydra configs for models and augmentations
│   ├── audio/            # Dataset adapters and audio preprocessing
│   ├── text/             # Dataset adapters and text cleaning
│   └── eval/results/     # Measured results
├── docs/
│   ├── architecture.md   # System design and planned structure
│   ├── api.md            # API contract for the planned service
│   ├── schema.md         # Database schema
│   ├── datasets.md       # Datasets, licences, sizes and evidence levels
│   ├── decisions.md      # Design decisions and the reasons behind them
│   └── datacards/        # Generated dataset datasheets
└── pyproject.toml        # Pytest configuration
```

---

## Setup and Installation

### Prerequisites

- Python 3.11
- A GPU for training. Free Colab or Kaggle sessions are enough
- ffmpeg, for the audio codec work

### 1. Clone the repository

```bash
git clone https://github.com/Shashank17singh/TruthLens.git
cd TruthLens
```

### 2. Install dependencies

Colab and Kaggle already include PyTorch, so skip the first `pip` line there.

```bash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r ml/requirements.txt
```

### 3. Run the tests

```bash
python -m pytest -m "not network"
```

### 4. Prepare a dataset

This small corpus downloads in seconds and runs the whole pipeline.

```bash
python -m ml.common.prepare detector_bias
```

Larger datasets use the same command. Some need a request form first, and the command prints instructions when files are missing. See [datasets](docs/datasets.md).

### 5. Train and evaluate

```bash
python -m ml.image.train model=convnext_tiny use_frequency=true
python -m ml.image.evaluate checkpoints/image/<run>/<run>-best.pt
python -m ml.image.export checkpoints/image/<run>/<run>-best.pt
python -m ml.image.results_table
```

Colab steps and the full experiment grid are in the [image detector guide](ml/image/README.md), and dataset details are in the [ML guide](ml/README.md).

---

## Roadmap

- [x] Dataset pipeline with leak-free splits
- [x] Image detector, evaluation and ONNX export
- [ ] Train the image models on real data
- [ ] Grad-CAM heatmaps and plain-language explanations
- [ ] Audio deepfake detector
- [ ] AI-generated text detector
- [ ] FastAPI backend with Celery workers
- [ ] React web app and Chrome extension
- [ ] Signed PDF reports
- [ ] Robustness and adversarial evaluation

---

## Known Limitations

- The image models have not been trained on real data, and no training run has used a GPU yet. Mixed precision and Weights & Biases logging are therefore untested.
- The image and audio adapters have only run on generated test files. Only the three text datasets above have been prepared for real.
- Most datasets predate current generators, so results will say little about the newest models until recent samples are added.
- MAGE publishes no prompt ids, so prompts are estimated from opening words. Topical and specified generations can still share a prompt across splits.
- Several datasets restrict use to non-commercial research. See [datasets](docs/datasets.md) before any public release.
- Results will be probabilistic and must not be used alone to make decisions about a person. Text detectors in particular can wrongly flag non-native English writers.

---
