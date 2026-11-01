# Design decisions

These record assumptions and choices made while designing TruthLens, including places where the feature prompts conflict or leave a gap.

## Architecture

### D1. Inference uses ONNX, explanations use PyTorch

Grad-CAM, spectrogram saliency, SHAP and integrated gradients need gradients, and ONNX Runtime does not provide them for this purpose. Workers therefore carry CPU PyTorch. ONNX produces the verdict, and PyTorch produces the explanation. `models.yaml` lists both files per version, and the export step checks that their outputs match within 1e-3.

Exporting only ONNX was rejected because it would limit explanations to methods without gradients, which conflicts with prompts 2.3, 2.4 and 2.5.

### D2. Separate Celery queues

A 5-minute audio clip produces 149 windows at 4 seconds with 50% overlap, so it can hold a worker for much longer than an image. Audio has its own queue so short jobs do not wait behind it. PDF rendering also has its own queue.

### D3. Models load after the worker forks

ONNX Runtime sessions should not be shared across forked processes. Each child process loads models once in `worker_process_init`, and thread counts are capped so processes do not compete for CPU cores.

### D4. Only Redis talks between workers and the API

Workers publish progress to Redis and API processes relay it to the WebSocket. This keeps working when several API processes run, because no worker needs to know which process holds the connection.

### D5. Training runs use Weights & Biases and Hydra

The pack allows either tracker. Colab and Kaggle sessions are temporary, so a local MLflow store would be lost with the session. Hydra handles the ablation runs through config overrides. Checkpoints are versioned in shared storage such as Google Drive, using names like `image-convnext-tiny-freq-1.0.0`.

## Conflicts and gaps in the prompts

### D6. The extension needs more than three permissions

Prompt 2.8 lists `contextMenus`, `storage` and `activeTab`. Injecting the overlay on demand with `chrome.scripting` also requires the `scripting` permission. The alternative, a content script declared for all sites, would ask for far broader access. Calling the backend also needs a `host_permissions` entry limited to the TruthLens API origin. Both are explained in the README for Chrome Web Store review.

### D7. The extension polls and the web app uses WebSocket

An MV3 service worker can be shut down while idle, which makes a long-lived WebSocket unreliable there. The extension polls `GET /analyses/{id}` with backoff.

### D8. Extra endpoints beyond prompt 2.6

The web app and report features require endpoints that prompt 2.6 does not list: accounts, media saving, API keys, admin, report creation, public report view and `/verify`. They are defined in [api.md](api.md).

### D9. "Check audio/video" analyses audio only

No prompt builds a video model. The extension's video option sends the source URL, and the worker extracts the audio track with ffmpeg. The result says that only the audio was checked. Direct media files work. Streaming pages and `blob:` URLs do not, and the extension says so.

### D10. Face and whole-image inputs

Prompt 2.1 crops faces only for face datasets, but a live upload has no such label. At inference, images with a detected face use the face crop, and other images use the whole frame resized to 224. Face-part explanations appear only when a face exists. Otherwise the explanation describes image regions. Results are also reported by dataset family, because one combined number could hide weak performance on either type.

### D11. Privacy applies to derived files too

A Grad-CAM overlay shows the uploaded face, and a sentence list contains the uploaded text, so the 24-hour rule covers both. Unless the user saves the analysis, the purge deletes uploads, artefacts, cached PDFs and stored sentences. A report requested later for a purged analysis returns `410`.

### D12. The report signature cannot be inside the PDF it signs

A file cannot contain the hash of its final bytes without changing them. The worker hashes the finished PDF, computes an HMAC over the hash and report id, and stores both in `reports`. `/verify` hashes the upload, finds the matching row and checks the HMAC in constant time. Anyone can use the page, but only the server can verify, because HMAC uses a shared secret.

### D13. The QR code shows a summary only

The QR link has to work for someone who is not logged in, so it uses an unguessable token and shows the verdict, model version and date. It never shows media.

### D14. Short text is accepted with a warning

Prompt 2.1 filters training text to 50–1,000 words, but prompt 2.5 tests short text and the extension will send short selections. The API accepts them and adds a warning to `warnings`, because the score is less reliable. Longer text uses sliding windows up to the 20,000-character limit.

### D15. Splits come in a fixed order

Prompt 2.1 asks for stratification by label and generator while also keeping identities, speakers and prompts apart. The steps are:

1. Remove held-out generators.
2. Assign whole groups (identity, speaker or prompt) to splits.
3. Stratify those groups as closely as the data allows.
4. Report the remaining imbalance.

Group separation always wins, since it is what prevents leakage.

Official splits are kept where a benchmark depends on them, such as ASVspoof.

## Risks

### D16. Dataset licences limit public release

FaceForensics++, GenImage and FakeAVCeleb are limited to non-commercial use. A free academic demo looks consistent with that, but the terms do not clearly cover a public API or a Chrome Web Store listing. Until the dataset authors confirm, keep the service free, label it academic, and record the training datasets for each model in `models.yaml`.

### D17. Reported accuracy may not hold on new generators

Most datasets here predate current image, voice and text models. The unseen-generator and cross-dataset results are the honest measure of generalization, and the paper should lead with them over in-dataset scores.

### D18. Text detection can harm people

False positives fall unfairly on non-native writers. The API always returns the disclaimer, the UI shows it next to the verdict, and the bias test results are reported whatever they show.

## Environment on the development machine

Checked on 2026-11-01:

| Item | State | Effect |
|---|---|---|
| Python 3.11 | Installed, but the default `python` is 3.14.2 | Create virtual environments with `py -3.11` |
| Node | 24.18.0 | Ready |
| Docker CLI and Compose | Installed, daemon not running | Start Docker Desktop before prompt 2.6 integration tests |
| ffmpeg | Missing | Install before prompt 2.4 codec work |
| NVIDIA GPU | `nvidia-smi` missing | Training stays on Colab or Kaggle |
