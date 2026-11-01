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

Preprocessing is the exception: it reads `ml/common/configs/preprocess.yaml` directly, because it has no sweeps.

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

### D19. A group may span `test` and `unseen_test`

Leakage means fitted data reaching evaluation. `test` and `unseen_test` are both evaluation splits, so a shared identity between them is harmless, and it lets a mixed face-swap identity contribute to both. A held-out sample in a group that reaches train or val is dropped and counted, so an unseen-generator score never benefits from a familiar face.

### D20. Audio is stored trimmed and cut at load time

Prompt 2.1 asks for 4-second clips. Storing one fixed clip per file would give training the same crop every epoch. Files are stored trimmed, training takes a random 4-second crop, and evaluation takes the first 4 seconds. Short audio is repeated, not padded with zeros, because a run of digital silence would be an easy false signal.

### D21. Silence is trimmed for scientific reasons too

Silence length correlates with the label in ASVspoof 2019, so a detector can score well without listening to the speech. Trimming removes that shortcut before any model sees the data.

### D22. RetinaFace comes from `batch-face`

`facenet-pytorch` pins PyTorch below 2.3 and NumPy below 2, which conflicts with Colab's current stack. `batch-face` provides RetinaFace with no such pins and also returns the five landmarks that prompt 2.3 can reuse.

### D23. Text is normalized before anything else

HC3's human answers are tokenized with spaces around punctuation and its ChatGPT answers are not. Without normalization, a model could reach high accuracy by reading spacing. All text passes through the same normalizer in training and in serving.

### D24. MAGE does not use its published split

The leakage check rejected MAGE's published split on the real data. After the length filter, 512 opening-word groups covering 13,144 rows (3.96%) span more than one split. One Reddit prompt alone has 142 texts across train and val, and some human texts recur with minor edits. These are real prompt overlaps, so a model would be evaluated on prompts it had already seen.

MAGE therefore uses the grouped 70/15/15 split, which is what prompt 2.1 asks for. The cost is that scores cannot be compared directly with the MAGE paper, and the paper must say so.

MAGE publishes no prompt ids. Grouping on the first 12 words links continuations to their human source, and it may join a few unrelated texts with the same stock opening, which is harmless. It cannot link topical or specified generations, because those do not reuse the human text. The LLaMA family is the unseen-generator set.

### D25. The frequency branch uses a DCT inside the model

The prompt allows FFT or DCT. An FFT in preprocessing would be a second code path to keep identical in training and serving, and FFT export to ONNX adds complex-number operators. A DCT is two matrix products, so it stays inside the model and exports as ordinary operators. The spectrum is computed from recovered pixels, so it means the same thing whichever normalization the backbone needs.

### D26. The frequency branch keeps spatial position

Position in a spectrum is frequency, and generator artefacts sit at particular frequencies. The branch ends in a 4×4 grid and not in global average pooling, which would discard where the energy is.

### D27. Late fusion happens at the logit level

Each branch produces its own logit and a learned weight combines them. This keeps each branch's contribution readable and makes the RGB-against-fusion ablation a direct comparison. Concatenating features was rejected because it hides how much each branch contributes.

### D28. Normalization follows the backbone

The ViT weights expect mean and std of 0.5, while EfficientNet and ConvNeXt expect ImageNet statistics. Values come from each model's timm configuration and are stored in the checkpoint and the export sidecar. Using one fixed normalization would silently degrade the ViT.

### D29. Calibration uses a temperature and a bias

Label smoothing trains the model to avoid confident outputs, and the balanced sampler trains it on a class mix that differs from validation. The first problem changes the scale of the logits and the second shifts them. A temperature alone cannot fix a shift: in a test run it left the classes separated around 0.41 instead of 0.5 and made ECE worse. Fitting a temperature and a bias on validation fixes both without changing the ranking, so AUC is unchanged. The API must return calibrated probabilities, because the confidence band and the 95% precision threshold both depend on them.

The temperature is bounded between 0.05 and 20. Without a bound, a model worse than chance sent it to 1.6e16, every probability became 0.5, and the report showed a perfect ECE of 0.000 for a broken model. Reports also carry a warning when validation AUC is 0.55 or lower, and `results_table` names those runs.

### D30. Nothing is tuned on test data

Calibration and the threshold come from validation only. A test flips every test label and confirms that both values stay identical. Precision depends on the share of fakes, so the 95% figure holds only near the validation mix and will be lower in a stream with far fewer fakes.

### D31. AUC intervals resample whole groups

Frames of one person are not independent, so a plain bootstrap would report an interval that is too narrow. The interval resamples identities instead, which is the honest uncertainty when two models are compared in the paper.

### D32. Training resumes after a session ends

Free Colab sessions end without warning. `last.pt` is written after every epoch through a temporary file, so a disconnect during a save cannot corrupt it, and re-running the same command continues from there.

### D33. ONNX parity is measured on images

A parity check on Gaussian noise failed a correct export. Noise is far outside the training data, so a small model returned logits near -1000, and at that size ordinary float32 rounding gave a difference of 0.2, or 0.02% of the logit. Layer-by-layer measurement showed the difference growing steadily through the network, with no single layer at fault. The check now uses validation images, or smooth generated images in the valid pixel range when no manifest is available. The 1e-3 tolerance did not change. The report records which inputs were used and the largest logit.

All six architecture variants, with ImageNet weights at 224×224, then passed with a largest difference of 2.3e-5. That shows the export itself is correct. The toy model's relative difference was larger than rounding in one operation would give, so the steady build-up through its layers is the best explanation available, and it was not proven separately. A trained model with unusually large logits on real images would still fail this check, and that failure should be investigated, not excused.

### D34. ONNX export traces with a batch of two

Tracing with a single image hard-codes the batch size into the graph even when the dimension is declared dynamic, and a later batch of two then crashes at a reshape. The trace uses two images, and verification runs batches of 1, 2 and 4.

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
