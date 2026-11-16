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

### D35. ViT is explained with attention rollout, not Grad-CAM

Grad-CAM needs a spatial feature map, and ViT has none until its tokens are reshaped into a grid, which is a guess about what the tokens mean. Attention rollout (Abnar and Zuidema, 2020) uses the model's own attention, averaged over heads, with the skip path added as an identity matrix. It is class-agnostic, so it shows where the model looked rather than what pushed it towards "fake". The UI must say which method produced a heatmap.

### D36. Regions are local maxima, not repeated argmax

The first version took the hottest pixel, boxed it, blanked the box and repeated. A test caught the flaw: the sloping edge left outside the box was reported as the next region. Local maxima found with a dilation filter cannot be an edge, and a peak inside an earlier box is skipped as a shoulder of that region.

### D37. Face parts come from RetinaFace's five landmarks

The detector already in the pipeline returns eyes, nose and mouth corners, so naming regions needs no second model. The hairline and chin are the top and bottom centre of the face box, and anything outside the box is background. Five points are coarse: a region between the nose and mouth is named after whichever is closer, and a cheek has no name of its own.

### D38. The two-second limit is a benchmark, not a unit test

On the development laptop with under 100 MB of free memory, one explanation took between 2 and 41 seconds as the system swapped, which says nothing about the code. The speed test runs only with `TRUTHLENS_PERF=1` on an idle machine, and the result belongs in the results folder with the machine named.

### D39. Audio features are computed inside the model

Log-mel, LFCC and CQT are fixed filterbanks written as convolutions, the same choice as the image DCT (D25). The model takes raw 16 kHz audio, so the server cannot compute features differently from training. The convolutional STFT matches librosa to within 1.2e-6 of the peak. The CQT starts at 50 Hz, because the default 32.7 Hz would need 8,000-sample filters, and speech carries little below 50 Hz.

### D40. One training loop for every modality

The image training loop moved to `ml/common/training.py`, with its loaders. Audio passes a hook that unfreezes the top transformer layers of wav2vec2 or WavLM after the frozen epochs. Every parameter is in the optimizer from the start, so unfreezing needs no new optimizer and resuming restores the same state.

### D41. Codecs go through ffmpeg, through a file

libsndfile can write MP3 and Opus, but it sets quality, not bitrate: its lowest MP3 setting gave 165 kbps at 16 kHz, so the prompt's 64 kbps was unreachable. ffmpeg encodes at an exact bitrate. The encoded clip goes to a temporary file, not a pipe, because ffmpeg writes the MP3 encoder delay into a header by seeking back, and without it the decoded clip was shifted. A test caught that shift.

### D42. Long audio reports mean and max, thresholded at the validation EER

Mean over windows suits a clip that is fake throughout. Max catches a short splice inside real speech but raises false alarms on long real recordings. Both are calibrated and reported. The decision threshold is the validation EER point, the usual operating point in anti-spoofing, and not the 95% precision target used for images.

### D43. min t-DCF only with the official ASV scores

The t-DCF needs the error rates of a speaker verification system working alongside the detector. Those come from the ASVspoof 2019 ASV score file, so min t-DCF is reported only when that file is given, and only for the ASVspoof 2019 evaluation set it belongs to.

EER, the ASV threshold and the legacy t-DCF follow the official `eval_metrics.py` from `asvspoof-challenge/2021` line by line, including its quirks: the DET threshold sits on a score and false alarms count with `>=`. Against that script on 50 random score sets, ties included, all three values matched exactly (largest difference 0.0). The first version interpolated the EER; it was close but would not have matched published numbers.

### D44. Audio windows are scored four at a time

Scoring all 14 windows of a 30-second clip in one batch made wav2vec2 fail with an allocation error on the 8 GB laptop. Four windows at a time keeps peak memory low on a small CPU server and costs the CNN models almost nothing. The measured latencies in `ml/eval/results/audio_latency.csv` were taken with only 1.2 to 2.4 GB of memory free, so they are pessimistic. wav2vec2 and WavLM take 10 to 30 seconds per 30-second clip on this CPU, so the deployed audio model is LCNN with LFCC, at about 0.5 seconds.

### D45. Binoculars with GPT-2 and DistilGPT-2

Binoculars was published with Falcon-7B and its instruct model, which do not fit an 8 GB CPU machine. GPT-2 small as the performer and DistilGPT-2 as the observer share one tokenizer, so their next-token distributions line up token by token, and both run on a CPU. Perplexity, burstiness, the GLTR rank buckets and entropy come from the same GPT-2 pass, so the eight features cost two forward passes per text. DetectGPT was left out: it needs dozens of perturbed rewrites per text, which is far beyond this budget.

### D46. Sentence scores and the confidence band reuse one pass

Per-token values are kept after the forward pass. A sentence's score is the meta-classifier applied to that sentence's tokens, which were predicted with the full left context. The 90% band is a bootstrap over sentences. Neither needs another model pass, and short texts get wide bands, which is the honest result for them.

### D47. Logistic regression as the meta-classifier

Eight features, plus one transformer logit when a fine-tuned model is available, do not need gradient boosting. A scaled logistic regression trains in seconds, its standardized weights say which signals drive a decision, and it adds no dependency. LightGBM can replace it if more features are added.

### D48. When a text is called mixed

A score between 0.35 and 0.65 is `mixed`. A text is also `mixed` when at least a quarter of its words sit in sentences scored as AI and a quarter in sentences scored as human, whatever the overall score. That second rule exists for human-edited AI text, where an average hides the split.

### D49. Transformer windows

Training takes one random 512-token window of each text per step, so long texts are seen in full over several epochs. Validation uses the first window, which keeps early stopping cheap. Final scoring averages the logits of all windows, which overlap by 128 tokens. Windows travel through the shared training loop as one tensor, ids stacked on the attention mask.

### D50. Integrated gradients by hand

Captum is not installed and integrated gradients is a short loop: 32 midpoint steps from a baseline of padding embeddings, with special tokens left unchanged. The completeness gap, the difference between the summed attributions and the logit change, is returned with every explanation so a bad approximation is visible.

### D51. Cross-dataset rows that open like training texts

MAGE groups texts by their first 12 words. HC3 and MAGE use different group ids, so the check hashes each HC3 text's first 12 words the same way and drops rows that match a MAGE train or validation group. On the prepared data this finds exactly the 497 rows the HC3 data card describes.

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
