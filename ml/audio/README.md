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

All three models were trained on a Kaggle T4 and evaluated on the official ASVspoof 2019 LA evaluation partition, scored as the mean logit over 4-second windows. Best validation AUC was 0.998 for the LCNN with LFCC, 0.9996 for wav2vec 2.0 and 0.9925 for WavLM.

| Model | Evaluation attacks | Clips | AUC (95% interval) | EER | min t-DCF |
|---|---|---|---|---|---|
| LCNN + LFCC | A16 and A19, the same algorithms as training attacks A04 and A06 | 17,182 | 0.992 (0.988 to 0.995) | 4.3% | 0.135 |
| | The 11 attacks never seen in training | 61,406 | 0.926 (0.915 to 0.937) | 15.2% | not computed |
| wav2vec 2.0 | A16 and A19 | 17,182 | 0.999 (0.998 to 0.999) | 1.4% | 0.041 |
| | The 11 unseen attacks | 61,406 | 0.960 (0.955 to 0.965) | 9.8% | not computed |
| WavLM | A16 and A19 | 17,182 | 0.931 (0.923 to 0.938) | 12.5% | 0.316 |
| | The 11 unseen attacks | 61,406 | 0.954 (0.949 to 0.959) | 10.7% | not computed |

EER per attack (%):

| Model | A07 | A08 | A09 | A10 | A11 | A12 | A13 | A14 | A15 | A16 | A17 | A18 | A19 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LCNN + LFCC | 7.83 | 0.04 | 0.00 | 34.62 | 13.72 | 26.65 | 15.61 | 0.65 | 5.23 | 5.05 | 9.18 | 18.76 | 3.29 |
| wav2vec 2.0 | 9.22 | 0.68 | 0.72 | 32.60 | 13.80 | 3.62 | 0.53 | 7.16 | 6.41 | 1.73 | 1.37 | 7.18 | 1.18 |
| WavLM | 10.50 | 1.69 | 1.16 | 32.07 | 6.25 | 4.07 | 1.20 | 8.12 | 7.94 | 5.31 | 7.22 | 17.38 | 18.80 |

- wav2vec 2.0 is the strongest model on both splits: EER 1.4% on known attacks and 9.8% on unseen ones, against 4.3% and 15.2% for the LCNN.
- WavLM does worse on the known attacks (12.5% EER) than on the unseen ones (10.7%). Most of its known-attack errors come from A19 (18.8% EER), so the validation set did not predict its test behaviour.
- A10 stays near 33% EER for every model, so no model handles that attack.
- The LCNN is still the one served. As ONNX it is 2.2 MB and scores a 30-second clip in 280 ms on four CPU threads; wav2vec 2.0 is 379 MB and takes 9.8 s.

Each model was trained once. Full output is in [ml/eval/results/audio/](../eval/results/audio/).

## Export and explain

```bash
python -m ml.audio.export checkpoints/audio/<run>/<run>-best.pt
python -m ml.audio.benchmark_exports --only lcnn
python -m ml.audio.explain checkpoints/audio/<run>/<run>-best.pt clip.flac
```

Export writes ONNX, or TorchScript if ONNX fails, and checks that its outputs match PyTorch. `ml/eval/results/audio_latency.csv` holds latencies for a 30-second clip measured on the development laptop. Explain saves a saliency map over time and frequency for the most suspicious window.
