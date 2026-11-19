# TruthLens: Multi-Modal Deepfake Detection & Forensic Attribution Platform

[![Vercel](https://img.shields.io/badge/Vercel-Live%20Demo-black?logo=vercel)](https://truth-lens-two-self.vercel.app/)
[![Hugging Face Space](https://img.shields.io/badge/Hugging%20Face-ZeroGPU%20Running-orange?logo=huggingface)](https://huggingface.co/spaces/shashank17singh/TruthLens)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org)
[![React 18](https://img.shields.io/badge/React-18-61DAFB?logo=react)](https://react.dev)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.x-3178C6?logo=typescript)](https://www.typescriptlang.org)

Detects AI-generated and manipulated images, audio, and text with explainable forensic artifacts (Grad-CAM heatmaps, 2D DCT frequency spectra), tamper-evident HMAC-SHA256 signed PDF audit trails, and multi-platform client applications.

> **Live Deployments:**
> * **Interactive Web Platform (Vercel):** [truth-lens-two-self.vercel.app](https://truth-lens-two-self.vercel.app/)
> * **ZeroGPU Model Space (Hugging Face):** [shashank17singh/TruthLens](https://huggingface.co/spaces/shashank17singh/TruthLens)

---

## Overview

Detectors that score well in isolated benchmarks often fail when exposed to unseen generation tools or aggressive social media compression. TruthLens addresses this generalization gap through multi-modal dual-branch architectures, frequency-domain artifact analysis, leak-free dataset split verification, and cryptographic report signing.

* **Vision Forensics:** Dual-branch ConvNeXt and EfficientNet backbones fused with 2D Discrete Cosine Transform (DCT) radial frequency analysis to detect upsampling checkerboard patterns and GAN/diffusion synthesis artifacts.
* **Acoustic Forensics:** Linear Frequency Cepstral Coefficients (LFCC) combined with Light-CNN (LCNN) architectures evaluated under ASVspoof standards for synthetic voice and replay attack detection.
* **Linguistic Forensics:** Multi-feature statistical ensemble combining GPT-2 token perplexity, burstiness metrics, GLTR ranking, and linguistic entropy scoring.
* **Audit Trail Integrity:** Automated generation of tamper-evident forensic PDF dossiers cryptographically signed with HMAC-SHA256 and embedded with dynamic QR codes for independent public verification.

Inference is optimized for both low-latency CPU deployment via ONNX Runtime and GPU acceleration via PyTorch on Hugging Face ZeroGPU.

---

## System Architecture

`mermaid
graph TD
    subgraph "Data Pipeline & Integrity"
    A[Public Datasets] -->|Checksum Verified| B(Dataset Adapters)
    B --> C{Preprocess & Face Crop}
    C --> D{Leak-Free Split Engine}
    D --> E[Manifests and Data Cards]
    end

    subgraph "Multi-Modal Inference Core"
    E --> F(RGB Spatial Backbone)
    E --> G(DCT Frequency Branch)
    F --> H{Late Fusion}
    G --> H
    H --> I[Temperature Calibration]
    I --> J[ONNX Export]

    E --> N(LFCC Audio Front-End)
    N --> O(LCNN Voice Classifier)
    O --> P[Sliding Window Aggregator]

    E --> Q(Token Perplexity & Burstiness)
    E --> R(GLTR Rank Distribution)
    Q --> S{Calibrated Ensemble}
    R --> S
    S --> T[Sentence Attribution]
    end

    subgraph "Production Services & Clients"
    J --> L[FastAPI Async Gateway]
    P --> L
    T --> L
    L --> U[(Celery Task Queue & Redis)]
    L --> V[HMAC-SHA256 Signed PDF Engine]
    L --> W[React 18 Cyber Forensics UI]
    L --> X[Chrome Extension Manifest V3]
    end

    classDef io fill:#f9f0ff,stroke:#8a2be2,stroke-width:2px,color:#000;
    classDef core fill:#e1f5fe,stroke:#0288d1,stroke-width:2px,color:#000;
    classDef logic fill:#e8f5e9,stroke:#388e3c,stroke-width:2px,color:#000;

    class A,E,J,P,T io;
    class B,F,G,I,N,O,Q,R,L,V core;
    class C,D,H,S,U,W,X logic;
`

---

## Core Capabilities

| Capability | Technical Implementation |
| :--- | :--- |
| **Dual-Branch Vision** | Fuses spatial RGB representations with DCT frequency representations computed directly in the tensor graph to capture grid artifacts. |
| **Explainable Heatmaps** | Grad-CAM, Grad-CAM++, and ViT attention rollout with automatic bounding-box facial landmark localization. |
| **Leak-Free Partitioning** | Automated verification engine guarantees complete identity, speaker, and prompt isolation across train, validation, and test splits. |
| **Unseen Generator Benchmarking** | Rigorous evaluation holding out entire model families (e.g., LLaMA, Midjourney, Stable Diffusion) from training. |
| **Calibrated Confidence** | Post-hoc temperature and vector scaling producing verifiable Bayesian reliability diagrams. |
| **Tamper-Evident Dossiers** | Asynchronous generation of multi-page PDF audit reports with embedded cryptographic HMAC-SHA256 signatures and QR verification. |
| **Client Ecosystem** | Production-ready React 18 single-page application with dark theme, real-time WebSocket progress, and Chrome Manifest V3 extension. |

---

## Benchmarks & Measured Results

### 1. CPU Inference Latency (ONNX Runtime)

Tested on single-thread Intel Core i5-10210U at 224x224 input resolution. All ONNX exports verified against native PyTorch tensors within 2.3e-5 tolerance:

| Architecture | Model Size | Spatial RGB Latency | Dual-Branch (RGB + DCT) |
| :--- | :--- | :--- | :--- |
| **EfficientNet-B0** | 18 MB | 27 ms | 45 ms |
| **ConvNeXt-Tiny** | 114 MB | 179 ms | 177 ms |
| **ViT-B/16** | 346 MB | 480 ms | 553 ms |

### 2. Audio Deepfake Detection (LCNN-LFCC)

Benchmarked on 30-second audio files evaluated over 4-second sliding windows with hop intervals:

| Model Architecture | Feature Representation | 1-Thread Latency | 4-Thread Latency |
| :--- | :--- | :--- | :--- |
| **LCNN (Deployed)** | LFCC (60 cepstral bins) | 533 ms | 280 ms |
| **ResNet-18** | LFCC | 940 ms | 640 ms |
| **wav2vec2** | Raw waveform | 30,530 ms | 9,755 ms |
| **WavLM** | Raw waveform | 21,809 ms | 10,915 ms |

### 3. Text Detection & Demographic Bias Mitigation

Evaluated across open-source synthetic benchmarks:

| Benchmark Corpus | Samples | ROC-AUC (95% CI) | Detection Rate |
| :--- | :--- | :--- | :--- |
| **MAGE Evaluation Split** | 800 | 0.768 (0.737 to 0.801) | 78.4% |
| **Unseen LLaMA Generation** | 500 | 0.812 (0.781 to 0.840) | 81.4% |
| **HC3 Cross-Domain** | 500 | 0.932 (0.910 to 0.952) | 91.8% |

---

## Monorepo Architecture

`
TruthLens/
├── backend/                  # FastAPI asynchronous server & Celery task worker
│   ├── app/
│   │   ├── api/              # REST & WebSocket route handlers (analyze, reports, verify)
│   │   ├── core/             # Configuration, security, JWT, rate limiting
│   │   ├── db/               # SQLAlchemy ORM models, SQLite / PostgreSQL session
│   │   ├── inference/        # Image (Grad-CAM, DCT), audio (LCNN), and text pipelines
│   │   ├── reports/          # Cryptographic PDF generation & HMAC verification
│   │   ├── storage/          # MinIO S3 & local filesystem fallback managers
│   │   └── worker/           # Celery asynchronous task definitions
│   └── tests/                # Pytest backend validation suite
├── frontend/                 # React 18 single-page application
│   ├── src/
│   │   ├── api/              # Strongly typed API client & hooks
│   │   ├── components/       # Visual gauges, spectrograms, modals, dropzones
│   │   ├── context/          # Auth and theme state providers
│   │   └── pages/            # Landing, analyze, result, history, verify, api-keys
│   └── vercel.json           # Client routing configuration
├── extension/                # Google Chrome Extension (Manifest V3)
│   ├── src/                  # Background service worker, content scripts, popup UI
│   └── manifest.json         # Extension permissions & declarative content scripts
├── ml/                       # Machine learning research and pipelines
│   ├── common/               # Checksum verifiers, leak-free splitters, data cards
│   ├── image/                # ConvNeXt, EfficientNet, ViT, Grad-CAM, DCT frequency
│   ├── audio/                # LCNN, LFCC extraction, ASVspoof scoring
│   ├── text/                 # Perplexity, burstiness, GLTR, RoBERTa classifiers
│   └── eval/                 # Robustness and adversarial evaluation suites
├── notebooks/                # GPU training workflows & ablation experiments
├── paper/                    # Academic artifacts
│   ├── main.tex              # IEEE conference paper source
│   ├── references.bib        # Comprehensive citation library
│   └── project_report.md     # 8-chapter university major project report
├── docs/                     # System documentation, API contracts, database schema
├── app.py                    # ZeroGPU Hugging Face Space entrypoint
└── pyproject.toml            # Workspace configuration
`

---

## Tech Stack

| Domain | Technologies |
| :--- | :--- |
| **Machine Learning** | Python 3.11, PyTorch, Torchvision, ONNX Runtime, timm, Transformers, Albumentations, Librosa, Scikit-learn |
| **Backend & APIs** | FastAPI, Uvicorn, Celery, Redis, SQLAlchemy, Pydantic v2, SQLite / PostgreSQL |
| **Storage & Security** | MinIO (S3-compatible), HMAC-SHA256, PyJWT, Passlib, WeasyPrint, ReportLab |
| **Frontend Web App** | React 18, TypeScript, Vite, Tailwind CSS, Lucide React, WaveSurfer.js, Vitest |
| **Browser Extension** | Chrome Manifest V3, TypeScript, Shadow DOM Injection |
| **Deployment** | Vercel (Edge Frontend), Hugging Face Spaces (ZeroGPU A10G), Docker Compose |

---

## Getting Started

### 1. Clone the Repository

`ash
git clone https://github.com/Shashank17singh/TruthLens.git
cd TruthLens
`

### 2. Set Up the Python Environment

`ash
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r ml/requirements.txt
`

### 3. Run the Backend API

`ash
# Start FastAPI application
python -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
`

Interactive documentation will be accessible at http://localhost:8000/docs.

### 4. Run the React Web Application

`ash
cd frontend
pnpm install
pnpm dev      # Starts Vite development server on http://localhost:3000
`

### 5. Build the Chrome Extension (Manifest V3)

`ash
cd extension
pnpm install
pnpm build    # Outputs compiled bundle to extension/dist/
`

To load in Chrome:
1. Open chrome://extensions/ in your browser.
2. Enable **Developer mode** in the top-right corner.
3. Click **Load unpacked** and select the TruthLens/extension/dist/ directory.

### 6. Verify Forensic PDF Reports

`ash
# Verify a downloaded PDF audit report via command line
curl -X POST http://localhost:8000/api/v1/verify \
  -F "file=@report.pdf"
`

### 7. Run Test Suites

`ash
# Run backend test suite
pytest backend/tests/ -v

# Run frontend test suite
cd frontend && pnpm test

# Run ML robustness validation
python -m ml.eval.robustness
`

---

## Roadmap

- [x] Dataset pipeline with leak-free splits
- [x] Multi-modal image detector, DCT frequency branch, and ONNX export
- [x] Calibrated benchmark models with CPU and ZeroGPU inference
- [x] Grad-CAM heatmaps and plain-language explainability
- [x] Audio deepfake detector with LFCC feature extraction
- [x] AI-generated text detector with linguistic entropy scoring
- [x] FastAPI backend with Celery asynchronous tasks
- [x] React web application with multi-modal explainability and verification
- [x] Chrome extension (Manifest V3) with Shadow DOM overlay
- [x] Cryptographically signed PDF forensic reports and HMAC verification
- [x] Robustness and adversarial evaluation suite
- [x] Research paper and academic documentation
- [x] Public ZeroGPU and edge web deployment

### Future Milestones

- [ ] Distributed multi-node training across full FaceForensics++ (1.8 TB) and DFDC video datasets
- [ ] Multi-GPU fine-tuning of 70B parameter LLM attribution decoders
- [ ] Live streaming audio forensic analysis via WebRTC media pipelines

---

## Known Limitations

- Production deployment runs on calibrated benchmark weights. Full-scale retraining on massive external video archives (such as multi-terabyte DFDC) requires a dedicated multi-GPU cluster.
- Image adapters have been verified on benchmark splits. ASVspoof 2019 LA and text datasets have been prepared with complete split integrity.
- Heavy foundation backbones (wav2vec2, WavLM, DeBERTa-v3) require high-VRAM GPU environments for end-to-end gradient updates.
- Paraphrase stress tests with large language models require substantial host memory.
- Deepfake detection is inherently probabilistic and intended to provide forensic assistance rather than automated punitive decisions.

---

## Citation

If you build upon or reference TruthLens in academic research:

`ibtex
@article{singh2026truthlens,
  title={TruthLens: Multi-Modal Deepfake Detection and Forensic Attribution with Explainable Artifacts and Tamper-Evident Verification},
  author={Singh, Shashank},
  year={2026}
}
`

---

## License

This project is licensed under the [MIT License](LICENSE).