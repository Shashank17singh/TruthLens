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

Re-running a command resumes from its last epoch. `steps_per_epoch` caps an epoch, which is useful on a CPU or inside a Colab session. The `robust` augmentation applies speed change, synthetic reverberation, coloured noise and an MP3 or Opus round trip at 64 kbps.

## Evaluate

```bash
python -m ml.audio.evaluate checkpoints/audio/<run>/<run>-best.pt
python -m ml.audio.evaluate checkpoints/audio/<run>/<run>-best.pt --asv-scores <ASVspoof2019.LA.asv.eval.gi.trl.scores.txt>
```

Long files are cut into 4-second windows with 50% overlap. The report gives both the mean and the max window logit, each calibrated on validation and thresholded at the validation equal-error point. It includes EER, AUC with a bootstrap interval by speaker, calibration error before and after, and EER per attack. min t-DCF appears only when the official ASV score file is given.

## Export and explain

```bash
python -m ml.audio.export checkpoints/audio/<run>/<run>-best.pt
python -m ml.audio.benchmark_exports --only lcnn
python -m ml.audio.explain checkpoints/audio/<run>/<run>-best.pt clip.flac
```

Export writes ONNX, or TorchScript if ONNX fails, and checks that its outputs match PyTorch. `ml/eval/results/audio_latency.csv` holds latencies for a 30-second clip measured on the development laptop. Explain saves a saliency map over time and frequency for the most suspicious window.
