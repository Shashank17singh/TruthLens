# Audio detector

Run every command from the repository root, after preparing `asvspoof2019_la` as described in the [ML guide](../README.md). ffmpeg must be on PATH for the MP3 and Opus augmentation.

Spectrogram models compute their features inside the network: log-mel, LFCC or CQT from 4-second clips at 16 kHz. Self-supervised models (wav2vec2 and WavLM) read the waveform.

## Train

On a GPU, [notebooks/colab_audio.ipynb](../../notebooks/colab_audio.ipynb) downloads the data, checks the archive, trains LCNN, wav2vec2 and WavLM with checkpoints on Drive, then evaluates and exports them.

```bash
python -m ml.audio.train model=lcnn feature=lfcc                  # the deployed CPU model
python -m ml.audio.train -m model=resnet18,lcnn feature=mel,lfcc,cqt
python -m ml.audio.train model=wav2vec2                           # GPU recommended
python -m ml.audio.train augment=none                             # ablation without augmentation
```

Re-running a command resumes from its last epoch. `steps_per_epoch` caps an epoch, which is useful on a CPU or inside a Colab session. `max_minutes=75` stops a run before the epoch that would overrun 75 minutes, so it ends cleanly inside a free Colab session; run the same command again to continue. The notebooks also run on Kaggle, whose Save & Run All mode has no idle limit. The `robust` augmentation applies speed change, synthetic reverberation, coloured noise and an MP3 or Opus round trip at 64 kbps.

## Evaluate

```bash
python -m ml.audio.evaluate checkpoints/audio/<run>/<run>-best.pt
python -m ml.audio.evaluate checkpoints/audio/<run>/<run>-best.pt --asv-scores <ASVspoof2019.LA.asv.eval.gi.trl.scores.txt>
```

Long files are cut into 4-second windows with 50% overlap. The report gives both the mean and the max window logit, each calibrated on validation and thresholded at the validation equal-error point. It includes EER, AUC with a bootstrap interval by speaker, calibration error before and after, and EER per attack. min t-DCF appears only when the official ASV score file is given.

## Results

The LCNN with LFCC was trained on a Kaggle T4 and stopped early after nine epochs, with a best validation AUC of 0.998. On the official ASVspoof 2019 LA evaluation partition, scored as the mean logit over 4-second windows:

| Evaluation attacks | Clips | AUC (95% interval) | EER | min t-DCF |
|---|---|---|---|---|
| A16 and A19, the same algorithms as training attacks A04 and A06 | 17,182 | 0.992 (0.988 to 0.995) | 4.3% | 0.135 |
| The 11 attacks never seen in training | 61,406 | 0.926 (0.915 to 0.937) | 15.2% | not computed |

Per-attack EER on the unseen attacks runs from 0.0% (A09) and 0.04% (A08) to 26.7% (A12) and 34.6% (A10). Full output is in [ml/eval/results/audio/](../eval/results/audio/).

wav2vec 2.0 and WavLM were trained too (best validation AUC 0.9996 and 0.9925), but their evaluation ran on CPU and did not finish inside Kaggle's 12-hour limit, so they have no evaluation-set results.

## Export and explain

```bash
python -m ml.audio.export checkpoints/audio/<run>/<run>-best.pt
python -m ml.audio.benchmark_exports --only lcnn
python -m ml.audio.explain checkpoints/audio/<run>/<run>-best.pt clip.flac
```

Export writes ONNX, or TorchScript if ONNX fails, and checks that its outputs match PyTorch. `ml/eval/results/audio_latency.csv` holds latencies for a 30-second clip measured on the development laptop. Explain saves a saliency map over time and frequency for the most suspicious window.
