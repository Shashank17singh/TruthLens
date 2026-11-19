# TruthLens: Explainable Multi-Modal Deepfake and Synthetic Content Detection Platform

## Major Project Report

---

### Executive Summary

TruthLens is an end-to-end multi-modal deepfake and AI-generated content detection platform developed for real-world forensic accountability. Modern generative AI architectures—spanning latent diffusion models, neural acoustic vocoders, and autoregressive large language models—create highly convincing synthetic media across visual, auditory, and textual modalities. Traditional forensic detectors are typically domain-siloed, report uncalibrated scalar scores without human-understandable evidence, fail under lossy social-media compression, and impose prohibitive GPU hardware constraints during inference.

TruthLens addresses these fundamental challenges through four architectural pillars:
1. **Multi-Modal Detection with Fine-Grained Explainability**: Fusing spatial ConvNeXt-Tiny with internal 2D Discrete Cosine Transform (DCT) frequency modeling and Grad-CAM region attribution for images; Light CNNs (LCNN) with Max-Feature-Map (MFM) activations over Linear Frequency Cepstral Coefficients (LFCC) for audio timeline scoring; and a stylometric meta-classifier combining token perplexity, burstiness variance, and GLTR rank histograms for text.
2. **Leak-Free Data Stratification**: Ensuring zero identity, speaker, or prompt leakage across training, validation, and held-out generator evaluation splits.
3. **Optimized CPU-First Inference**: Compiling models to ONNX Runtime achieving sub-300 ms latencies on commodity multi-core processors.
4. **Full-Stack Forensic Integration**: Providing an asynchronous FastAPI and Celery worker engine, a React 18 dashboard, a Manifest V3 browser extension with Shadow DOM isolation, and cryptographically signed, tamper-evident PDF reports verifiable via HMAC-SHA256.

---

## Chapter 1: Introduction

### 1.1 Background and Motivation
The emergence of generative artificial intelligence has altered the information ecosystem. Deepfake imagery, cloned voices, and synthetic prose can be generated instantaneously with open-source tools. When deployed maliciously, synthetic media enables election interference, corporate executive fraud, financial voice-phishing, cyber-harassment, and academic dishonesty. Forensic investigators require tools that not only detect manipulation with high precision but also explain *why* content was flagged, withstand network compression, and provide auditable cryptographic proof of verification.

### 1.2 Problem Statement
Existing synthetic media detectors suffer from four critical deficiencies:
1. **Single-Modality Bias**: Detection tools specialize in either video, audio, or text, leaving multi-modal forgery campaigns undetected.
2. **Black-Box Opacity**: Scalar confidence values (e.g., "94% Fake") provide no forensic rationale, rendering decisions inadmissible in legal or academic proceedings.
3. **Distribution Shift Fragility**: Detectors trained on pristine, uncompressed benchmarks suffer severe accuracy degradation ($>20\%$) when evaluated on social media platforms (WhatsApp, X/Twitter) or novel generative models.
4. **Prohibitive Hardware Requirements**: Many published models rely on massive transformer ensembles requiring dedicated GPU clusters, preventing ubiquitous desktop and edge deployment.

### 1.3 Project Objectives
The primary objectives of TruthLens are:
- Build an integrated multi-modal platform capable of detecting synthetic images, audio, and text.
- Engineer explainability features: Grad-CAM saliency overlays with facial landmark bounding boxes for images, temporal anomaly timelines for audio, and per-sentence token entropy highlights for text.
- Ensure all models execute on standard multi-core CPUs in under 500 ms per standard query via ONNX Runtime optimization.
- Prevent dataset leakage by strictly isolating human identities, voice speakers, and prompts across splits.
- Evaluate model robustness against compression, additive noise, adversarial perturbations (FGSM and PGD), and cross-dataset domain shifts.
- Deliver an asynchronous backend (FastAPI, Celery, Redis), an interactive web app, a browser extension, and tamper-verifiable PDF reports.

### 1.4 Scope and Constraints
- **In-Scope**: Facial manipulation and synthetic portrait imagery, speech cloning and vocoder synthesis, AI-generated English prose, asynchronous batching, browser extension integration, and cryptographic document signing.
- **Out-of-Scope**: Non-speech biological acoustic classification, real-time video stream transcoding beyond extracted frames, and non-English text detection.

### 1.5 Report Organization
This report is structured into eight chapters: Chapter 2 surveys related literature; Chapter 3 outlines requirements and system architecture; Chapter 4 describes the data engineering and leak-free split methodology; Chapter 5 details multi-modal model designs and explainability mechanisms; Chapter 6 explains backend, frontend, extension, and PDF implementation; Chapter 7 presents empirical benchmarks, robustness evaluations, and adversarial findings; Chapter 8 concludes with ethical considerations, limitations, and future directions.

---

## Chapter 2: Literature Survey and Related Work

### 2.1 Visual Deepfake Detection
Early visual deepfake detection focused on handcrafted physiological markers such as abnormal blink frequencies (Li et al.) and irregular head pose dynamics. Modern deep learning approaches shifted toward convolutional backbones, notably EfficientNet and ConvNeXt (Liu et al., 2022). Frank et al. (2020) demonstrated that transposed convolutions and bilinear upsampling in generative networks leave persistent periodic grid artifacts in the 2D Fourier and Discrete Cosine Transform (DCT) domains. TruthLens combines spatial representation learning with an integrated 2D DCT branch.

### 2.2 Acoustic Anti-Spoofing and Voice Cloning
The biennial ASVspoof challenges (Todisco et al., 2019) benchmarked voice anti-spoofing against logical access (LA) synthesis and voice conversion attacks. High-performing baselines utilize front-end representations including Linear Frequency Cepstral Coefficients (LFCC) and Constant Q-Transform (CQT) features fed into Light CNNs (LCNN) with Max-Feature-Map (MFM) activations (Wu et al., 2018). While self-supervised backbones (wav2vec 2.0, WavLM) exhibit strong representation capacity, their model footprints (>350 MB) incur substantial CPU latencies (>9 seconds for 30s clips). LCNN offers an optimal trade-off: 2.2 MB size and 280 ms execution.

### 2.3 Synthetic Text Attribution
Statistical text attribution analyzes statistical signatures inherent to autoregressive decoding (temperature, top-$k$, top-$p$ nucleus sampling). GLTR (Gehrmann et al., 2019) tracks predictive token rank distributions. DetectGPT (Mitchell et al., 2023) and Binoculars (Hans et al., 2024) evaluate likelihood curvature and perplexity ratios across model pairs. Liang et al. (2023) demonstrated that statistical text detectors systematically exhibit elevated false positive rates against non-native English writers, making calibrated confidence intervals and forensic disclaimers necessary.

### 2.4 Explainability in Digital Forensics
Explainable AI (XAI) is critical for forensic verification. Saliency methods such as Grad-CAM (Selvaraju et al., 2017) compute gradient-weighted feature map activations. Adebayo et al. (2018) established sanity checks showing that many edge-based saliency methods function as edge detectors independent of learned weights; Grad-CAM successfully passes weight-randomization tests.

---

## Chapter 3: System Requirements and Architecture

### 3.1 Requirements Analysis
- **Functional Requirements**:
  - Upload and analyze media via files or external URLs.
  - Asynchronous background job processing with WebSocket progress streaming.
  - Multi-modal results view with localized visual, acoustic, and text evidence.
  - History logging, user feedback loop, and API key management with tiered rate limiting.
  - Downloadable signed PDF reports and public tamper verification endpoint.
- **Non-Functional Requirements**:
  - Sub-500 ms CPU inference per single-item analysis.
  - Zero data leakage between evaluation and training sets.
  - Ephemeral 24-hour media storage for user privacy.
  - Accessible, responsive UI compatible with screen readers and mobile viewports.

### 3.2 Monorepo Structure
```text
TruthLens/
├── backend/            # FastAPI, Celery tasks, SQLite/PostgreSQL, MinIO client, reports
├── frontend/           # React 18, TypeScript, Vite, Tailwind CSS, TanStack Query
├── extension/          # Chrome Extension Manifest V3, Shadow DOM overlay
├── ml/
│   ├── common/         # Leak-free splitting, data cards, training harness
│   ├── image/          # ConvNeXt, DCT branch, Grad-CAM, ONNX export
│   ├── audio/          # LCNN, LFCC extraction, windowed inference
│   ├── text/           # Perplexity, burstiness, GLTR, logistic ensemble
│   └── eval/           # Multi-modal robustness, FGSM/PGD attacks, IEEE figures
├── notebooks/          # Colab GPU training and evaluation walkthroughs
└── paper/              # IEEE conference paper (LaTeX, BibTeX, figures)
```

### 3.3 Database Schema
The relational schema comprises six core entities:
1. `users`: ID (UUIDv7), email, hashed password, role, timestamps.
2. `analyses`: Analysis ID, user foreign key, modality (`image`, `audio`, `text`), status (`queued`, `processing`, `completed`, `failed`), file metadata, SHA-256 digest, privacy retention flag, expiration timestamp.
3. `analysis_results`: Result ID, analysis foreign key, probability, calibrated threshold, confidence intervals, verdict (`likely_fake`, `uncertain`, `likely_real`), explanation string, model version, JSON details (bounding boxes, audio windows, sentence spans).
4. `reports`: Report ID, analysis foreign key, public URL token, status, object key, SHA-256 digest, HMAC-SHA256 signature, signing key ID.
5. `api_keys`: Key hash, prefix, user foreign key, rate limits, revocation status.
6. `feedback`: Analysis foreign key, user foreign key, correctness boolean, reported label, user comments.

---

## Chapter 4: Data Engineering and Leak-Free Pipeline

### 4.1 Datasets Utilized
- **Image**: FaceForensics++ (1,000 video sequences, 4 manipulation algorithms, multiple compression rates), Celeb-DF-v2, and WildDeepfake.
- **Audio**: ASVspoof 2019 Logical Access (107,180 utterances, 19 speech synthesis/conversion algorithms, 107 human speakers).
- **Text**: MAGE and HC3 datasets covering human and machine text across news, academic essays, and technical documentation.

### 4.2 Leak-Free Partitioning Algorithm
To prevent data leakage, stratification strictly isolates biological identities and generation prompts:
1. **Faces (Image)**: Split strictly by actor identity across video sequences. If subject ID `042` appears in training, no frame of subject `042` may appear in validation or testing.
2. **Voices (Audio)**: Split strictly by speaker ID. In addition, unseen spoofing algorithms (attacks A10--A19) are held out exclusively for test splits.
3. **Text**: Split strictly by source prompt and domain to prevent memorization of repetitive topic tokens.
4. **Automated Leak Checks**: Automated unit tests assert zero set intersection across split manifests before training scripts execute.

---

## Chapter 5: Multi-Modal Model Development & Explainability

### 5.1 Image Detector (ConvNeXt-Tiny + 2D DCT Frequency Branch)
- **RGB Backbone**: Pre-trained ConvNeXt-Tiny extracting 768-dimensional spatial representations.
- **DCT Branch**: On-the-fly luminance calculation followed by matrix-multiplied orthonormal 2D DCT-II:
  $$\text{Spectrum} = \log(|B \cdot Y \cdot B^T| + \delta)$$
  Fed into a 4-layer convolutional network preserving spatial frequency grid coordinates.
- **Late Fusion**: Linear combination layer combining spatial and spectral logits: $z_{\text{fused}} = w_1 z_{\text{rgb}} + w_2 z_{\text{freq}}$.
- **Explainability**: Grad-CAM saliency extraction mapped against facial landmark regions (ocular, oral, nose, boundary) yielding natural-language attribution.

### 5.2 Audio Deepfake Detector (LCNN + LFCC)
- **Acoustic Front-End**: 60-dimensional Linear Frequency Cepstral Coefficients (LFCC) capturing vocoder harmonic phase anomalies.
- **Classifier**: Light CNN (LCNN) with Max-Feature-Map (MFM) activations: $\text{MFM}(x)_k = \max(x_k, x_{k + C/2})$.
- **Windowed Inference**: 4.0-second sliding windows with 50% overlap; aggregated via dual mean-max pooling to output a localized timeline track.

### 5.3 Text Detector (Stylometric Ensemble)
- **Perplexity & Burstiness**: Token likelihood evaluated under GPT-2 Small. AI text exhibits low perplexity and minimal sentence-to-sentence burstiness variance.
- **GLTR Histogram**: Proportion of tokens within top-10, top-100, and tail ranks.
- **Ensemble Meta-Classifier**: Regularized logistic regression combining statistical stylometry with transformer log-probabilities, producing a calibrated score and 95% bootstrap confidence intervals.

---

## Chapter 6: Production Engineering and Implementation

### 6.1 Asynchronous Backend (FastAPI + Celery + Redis + MinIO)
- Submissions receive an immediate `202 Accepted` response with an analysis ID.
- Celery worker pools load ONNX models into memory once per worker process.
- Progress events are published over Redis pub/sub channels and broadcast to clients via WebSockets.
- File validation enforces MIME magic-byte checks, EXIF metadata stripping, and 24-hour automatic storage purge.

### 6.2 Modern Web Dashboard (React 18 + TypeScript + Tailwind)
- Clean, responsive UI with dedicated tabs for Image, Audio, Text, and URL inputs.
- Real-time probability gauge with color-blind safe palettes (green, amber, rose).
- Interactive visualizers: Grad-CAM overlay opacity slider, WaveSurfer audio timeline player, and sentence-level text highlight spans.

### 6.3 Chrome Extension (Manifest V3)
- Context menu integration: "Check image with TruthLens" and "Check selected text".
- Shadow DOM overlay injection isolating forensic result panels from host webpage CSS.
- Background service worker utilizing user API keys stored in `chrome.storage.local`.

### 6.4 Cryptographic PDF Verification
- Headless PDF generation via WeasyPrint with pure-Python ReportLab fallback.
- Server-side HMAC-SHA256 signature over final document bytes:
  $$\text{Signature} = \text{HMAC}_{\text{key}}(\text{SHA256}(\text{PDF\_Bytes}))$$
- Embedded dynamic QR code linking to verification portal (`https://truthlens.ai/verify?report_id={id}`).
- Verification endpoint performing constant-time hash comparisons for tamper detection.

---

## Chapter 7: Experimental Results, Robustness & Adversarial Evaluation

### 7.1 Baseline Classification Performance
| Modality | Architecture | Accuracy | ROC-AUC | EER / Error |
| :--- | :--- | :---: | :---: | :---: |
| **Image** | ConvNeXt-Tiny + DCT Frequency | **94.6%** | **0.981** | 5.32% |
| **Audio** | LCNN + LFCC (30s clip) | **95.2%** | **0.988** | **2.14%** |
| **Text** | Stylometric Logistic Ensemble | **93.5%** | **0.972** | 4.80% FP |

### 7.2 CPU Serving Latency Benchmarks (Intel Core i5)
- **ConvNeXt-Tiny (ONNX)**: 177.3 ms (1-thread) $\rightarrow$ 93.4 ms (4-threads).
- **LCNN Audio (ONNX)**: 533.2 ms (1-thread) $\rightarrow$ 280.2 ms (4-threads for 30s audio).
- **wav2vec2 Comparison**: wav2vec2 requires 9,754 ms on 4 threads; LCNN is **35x faster** while fitting within 2.2 MB.

### 7.3 Robustness under Social Media Channel Noise
- **Lossy JPEG**: Accuracy drops from 94.6% (clean) to 88.7% ($Q=70$) and 82.4% ($Q=50$), demonstrating high-frequency DCT coefficient attenuation.
- **Social Media Pipeline**: Compound compression (downscale + noise + JPEG 65) reduces accuracy to 78.6%.
- **Audio Phone-Band (300--3400 Hz)**: Filters out high-frequency vocoder cues, increasing EER from 2.14% to 14.85%.
- **Text Paragraph Mixing**: 50% human mixing degrades AI detection rate from 93.5% to 58.2%.

### 7.4 Adversarial Attacks (FGSM & PGD) and Defense
- **Unhardened Model**: Under PGD attacks at $\epsilon = 8/255$, standard ConvNeXt accuracy collapses from 94.6% to 3.86%.
- **Adversarially Hardened Model**: Fast adversarial training retains 56.8% (FGSM) and 38.08% (PGD) accuracy at $\epsilon = 8/255$, incurring a minor 2.8% drop in clean accuracy (94.6% $\rightarrow$ 91.8%).

### 7.5 Cross-Dataset Generalization Gaps
- Image: FF++ $\rightarrow$ DiffusionFace: 24.4% ROC-AUC drop.
- Audio: ASVspoof 2019 $\rightarrow$ In-The-Wild Audio: 18.9% ROC-AUC drop.
- Text: MAGE/HC3 $\rightarrow$ Non-Native English Essays: 23.3% ROC-AUC drop due to vocabulary and syntax distribution shifts.

---

## Chapter 8: Conclusion, Limitations and Future Work

### 8.1 Summary of Contributions
TruthLens delivers a unified, production-grade synthetic media attribution platform. By coupling spatial-frequency modeling, compact acoustic CNNs, and stylometric ensembles with optimized ONNX CPU inference, TruthLens proves that multi-modal forensic transparency is achievable on standard computing infrastructure.

### 8.2 Limitations
1. **Bias Against Non-Native Writers**: Stylometric detectors misclassify essays written by non-native English speakers due to lower burstiness variance.
2. **Heavy Compression False Negatives**: Severe lossy re-encoding destroys fine spectral cues, leading to elevated false negative rates.
3. **Emerging Generative Paradigms**: Detectors trained on contemporary datasets require continuous updates as novel generative architectures emerge.

### 8.3 Future Work
- Incorporating vision-language cross-attention transformers for contextual multimodal narrative checking.
- Exploring zero-watermarking and cryptographic provenance standards (C2PA).
- Implementing federated fine-tuning for continuous forensic model adaptation without centralized media storage.

---

## 15 Likely Viva Questions and Model Answers

**Q1: What is the core motivation for using a dual-branch (RGB + DCT) architecture in the image detector?**
> **Answer**: Deepfake generators (both GANs and Diffusion models) employ upsampling layers (such as transposed convolutions or nearest-neighbor interpolation) that leave subtle, periodic grid artifacts. While convolutional filters in the RGB domain primarily capture spatial semantic features, an explicit 2D Discrete Cosine Transform (DCT) decomposes luminance into orthonormal frequency coefficients, directly exposing high-frequency spectral irregularities that are invariant to localized lighting and color variations.

**Q2: What is data leakage in synthetic media detection, and how does TruthLens prevent it?**
> **Answer**: Data leakage occurs when samples sharing the same identity (faces), speaker (audio), or prompt (text) appear in both training and evaluation splits. Models can memorize facial morphology or vocal timbres rather than learning generalizable synthetic artifacts, resulting in inflated evaluation metrics that collapse in production. TruthLens implements strict stratification: entire identities, vocal speakers, and prompts are assigned to single splits, and held-out generator splits test zero-shot generalization.

**Q3: Why did you choose LCNN with Max-Feature-Map (MFM) activations for audio instead of a large transformer like wav2vec2 or Whisper?**
> **Answer**: Wav2vec 2.0 requires ~380 MB of memory and takes over 9.7 seconds to process a 30-second audio clip on a 4-thread CPU. In contrast, the LCNN architecture with MFM activations occupies only 2.2 MB and executes in 280 ms (over 35x faster), while achieving a state-of-the-art 2.14% Equal Error Rate (EER) on ASVspoof 2019. MFM acts as an intrinsic noise filter by retaining only the maximum activation between feature map pairs, preserving vocoder phase discontinuities efficiently.

**Q4: How does the text detector work, and why is perplexity alone insufficient?**
> **Answer**: Perplexity measures how expected each token is under an autoregressive language model (GPT-2). However, human writers can also write low-perplexity sentences when discussing common topics. TruthLens evaluates *burstiness*—the variance of perplexity across sentences—alongside GLTR predictive rank histograms and supervised embeddings. Human prose exhibits high burstiness (mixing short, simple sentences with complex clauses), whereas LLMs generate sentences with uniform perplexity profiles.

**Q5: Why do text detectors exhibit higher false positive rates on non-native English writing?**
> **Answer**: Non-native English writers often rely on simpler, standard vocabulary, formulaic sentence structures, and textbook transitional phrases. This reduces lexical diversity and sentence-to-sentence burstiness variance, mirroring the statistical profile of large language models. Research (Liang et al., 2023) confirmed this bias. TruthLens addresses this by incorporating wide confidence intervals and a mandatory forensic disclaimer prohibiting punitive action based solely on automated scores.

**Q6: What is Equal Error Rate (EER), and how is it calculated for the audio detector?**
> **Answer**: Equal Error Rate (EER) is the biometric evaluation metric where the False Acceptance Rate (FAR—synthetic speech classified as human) equals the False Rejection Rate (FRR—genuine human speech classified as synthetic). It is computed by finding the decision threshold $\theta^*$ on the Receiver Operating Characteristic (ROC) curve where $\text{FAR}(\theta^*) = \text{FRR}(\theta^*)$. Lower EER indicates superior discrimination. TruthLens achieves 2.14% EER on ASVspoof 2019.

**Q7: Explain the difference between FGSM and PGD adversarial attacks.**
> **Answer**: Fast Gradient Sign Method (FGSM) is a single-step attack that perturbs the input image in the direction of the loss gradient: $x_{\text{adv}} = \text{clamp}(x + \epsilon \cdot \text{sign}(\nabla_x \mathcal{L}), 0, 1)$. Projected Gradient Descent (PGD) is an iterative multi-step attack: it starts from a random perturbation within the $L_\infty$ $\epsilon$-ball and repeatedly takes smaller gradient steps ($\alpha = \epsilon / 4$), projecting back onto the feasible set at each step. PGD finds significantly stronger adversarial examples, causing unhardened ConvNeXt accuracy to drop to 3.86% at $\epsilon = 8/255$.

**Q8: How does adversarial training improve model robustness against adversarial perturbations?**
> **Answer**: Adversarial training solves a robust min-max optimization problem: $\min_\theta \mathbb{E}_{(x, y)} [\max_{\delta \in \mathcal{S}} \mathcal{L}(f_\theta(x + \delta), y)]$. During training, each mini-batch includes both clean images and worst-case PGD-perturbed images. The network parameters $\theta$ are updated to minimize the loss on both. This flattens the loss landscape around training points, raising PGD accuracy from 3.86% to 38.08% at $\epsilon = 8/255$.

**Q9: Why does lossy JPEG compression disproportionately impact the frequency branch?**
> **Answer**: JPEG compression segments an image into $8 \times 8$ pixel blocks, applies a 2D DCT, and divides the resulting frequency coefficients by a quantization matrix before Huffman entropy coding. High-frequency coefficients (which correspond to fine details and subtle generative upsampling grid artifacts) are aggressively quantized to zero. At low quality factors ($Q \le 50$), the spectral fingerprints relied upon by the frequency branch are eradicated, dropping accuracy from 94.6% to 82.4%.

**Q10: What is the purpose of the 2D DCT matrix formulation in `DctSpectrum` compared to an FFT?**
> **Answer**: Implementing the 2D DCT as fixed matrix multiplication ($D = B \cdot Y \cdot B^T$) allows the frequency transformation to reside directly inside the PyTorch computational graph using standard matrix products. This ensures the model exports cleanly to standard ONNX runtimes without requiring custom C++ operators or external FFT library dependencies, guaranteeing identical inference behavior between Python and ONNX.

**Q11: How is tamper-evidence guaranteed in the downloadable PDF reports?**
> **Answer**: When a report is rendered, TruthLens computes the SHA-256 hash of the complete PDF byte array and signs it using HMAC-SHA256 with a private server secret. The signature and hash are recorded in the database and printed on the document alongside a QR code. When a document is uploaded to `/api/v1/verify`, the backend recomputes the SHA-256 hash and verifies it against the HMAC in constant time (`hmac.compare_digest`). Any modification to text, scores, or timestamps alters the hash and invalidates the signature.

**Q12: Why did you use Celery and Redis for asynchronous task execution rather than standard FastAPI background tasks?**
> **Answer**: FastAPI `BackgroundTasks` run in the same process and event loop as the web server, which can block HTTP request handling during CPU-intensive inference or PDF rendering. Celery distributes tasks across dedicated worker processes that can be scaled across multiple machines. Redis acts as a high-throughput message broker and pub/sub channel, allowing workers to push live inference progress updates to connected clients over WebSockets.

**Q13: How does the Chrome extension handle CORS restrictions when analyzing web images?**
> **Answer**: Many external domains serve images with strict Cross-Origin Resource Sharing (CORS) headers that prevent browser scripts from reading canvas pixel data directly. The TruthLens extension circumvents this by sending the verified media source URL directly to the backend (`POST /api/v1/analyze/url`). The backend downloads the media stream server-side, validates its MIME type and magic bytes, strips EXIF data, and initiates inference without CORS restrictions.

**Q14: What is the sanity check for saliency maps (Adebayo et al.), and why did you run it on Grad-CAM?**
> **Answer**: Adebayo et al. (2018) discovered that several popular saliency methods (such as Guided Backpropagation) produce visually pleasing edge maps that remain virtually unchanged even when the neural network's weights are completely randomized, acting like edge filters rather than model explanations. TruthLens runs the model-randomization test by comparing Grad-CAM activations before and after cascading layer randomization. The Grad-CAM heatmap completely dissolves when weights are randomized, confirming that the visual explanation reflects learned model weights.

**Q15: What privacy measures are implemented for user-uploaded media?**
> **Answer**: Deepfake detection platforms often handle sensitive personal photos and voice clips. TruthLens enforces data minimization: upon upload, all EXIF metadata and GPS coordinates are stripped before disk write. By default, uploaded media and derived heatmaps are marked with an ephemeral 24-hour expiration timestamp. A periodic Celery purge task automatically deletes expired media from MinIO and database storage unless the user explicitly toggles the `retain_media` flag.
