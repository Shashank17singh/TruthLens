# Datasets

Checked on 2026-11-01 from web search results, not from each dataset's full licence text. Read the licence yourself before downloading or citing a dataset. The Evidence column shows how far each row can be trusted:

- **Official**: seen on the dataset's own page, repository or record.
- **Secondary**: seen only in a mirror, a shared-task table or an announcement.
- **Unconfirmed**: not found, or recalled from memory. Do not cite it yet.

## Apply now

These need a form or a terms agreement, and approval can take days:

| Dataset | What to do |
|---|---|
| FaceForensics++ | Follow the instructions in the official repository. Sources disagree on whether this is a Google form or a signed agreement sent by email. |
| Celeb-DF v2 | Fill in the Google form linked from the official repository. |
| FakeAVCeleb (optional) | Use a college email and write a research description of at least 100 words. Access expires after 1 week, so be ready to download at once. |
| DFDC (optional) | Accept the terms on the official page. |

## Image

| Dataset | Size | Generators or content | Licence | Request form | Evidence | Role |
|---|---|---|---|---|---|---|
| [FaceForensics++](https://github.com/ondyari/faceforensics) | About 10 GB at c23 (unconfirmed) | 1,000 source videos plus face-manipulation methods | FaceForensics Terms of Use: non-commercial research and education only | **Yes** | Official for terms | **Primary** for face manipulation |
| [Celeb-DF v2](https://github.com/yuezunli/celeb-deepfakeforensics) | 590 real and 300 extra YouTube videos, 5,639 fake; 12.4 GB on a mirror | Improved face swap | Custom Celeb-DF terms, not read | **Yes** | Official for counts and form; secondary for size | **Primary** cross-dataset test |
| [GenImage](https://genimage-dataset.github.io) | 2,681,167 images; download size not found | 8: ADM, BigGAN, Midjourney, VQDM, GLIDE, SD 1.4, SD 1.5, Wukong | Non-commercial; a mirror states CC BY-NC-SA 4.0 | No, but official downloads use Baidu Yunpan | Official for non-commercial; secondary for licence name | **Primary** for whole-image synthesis, using a subset |
| [ArtiFact](https://arxiv.org/abs/2302.11970) | 2,496,738 images | 25 methods: 13 GAN, 7 diffusion, 5 other | Not found | Not found | Secondary | Optional cross-dataset test |
| [DiffusionForensics](https://github.com/ZhendongWang6/DIRE) | 439,020 fake, 92,000 real | Diffusion models | Not found | Not found | Secondary | Optional |
| [DFDC](https://ai.facebook.com/datasets/dfdc) | Over 100,000 clips from 3,426 paid actors; about 470 GB (unconfirmed) | Several face-swap methods | Not found | Terms to confirm | Secondary | Optional, and only as a frame subset |
| [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) | 120,000 images at 32×32 | SD 1.4 | Conflict: MIT in most listings, CC BY-NC 4.0 in one | No | Secondary | **Pipeline smoke tests only** |

CIFAKE images are 32×32, so upscaling them to 224×224 tests the resize and not the forgery. Its numbers should not appear in the paper.

## Audio

| Dataset | Size | Generators or content | Licence | Request form | Evidence | Role |
|---|---|---|---|---|---|---|
| [ASVspoof 2019 LA](https://datashare.ed.ac.uk/handle/10283/3336) | 7.116 GB | TTS and VC attacks A01–A19 | Open Data Commons Attribution | No | Official | **Primary**, and the source for min t-DCF |
| [ASVspoof 2021 DF](https://zenodo.org/record/4835108) | 34.5 GB in 4 parts; the `eval` phase has 14,869 bona fide and 519,059 spoof trials | TTS and VC with codecs | Conflict: Zenodo metadata says Open Database License, a 2022 announcement says Attribution | No | Official for size, counts and Zenodo licence | **Primary** evaluation set; it has no training split, and its keys are a separate download |
| [WaveFake](https://zenodo.org/record/5642694) | 104,885 clips, 28.9 GB | GAN vocoders on LJSpeech and JSUT | CC BY-SA 4.0 | No | Official | Per-vocoder evaluation |
| [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) | 2.75 GB; 13,100 clips (unconfirmed) | Real speech from one speaker | Public domain (unconfirmed) | No | Official for size | Real side of WaveFake |
| [In-the-Wild](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild) | 20.8 h real, 17.2 h fake, 58 speakers | Public-figure recordings | Apache 2.0 on the official page; a mirror says CC BY-SA 4.0 | No | Official | **Primary** cross-dataset test |
| [ASVspoof 5](https://zenodo.org/records/14498691) | 142.3 GB | Crowdsourced speech, newer attacks | Open Data Commons Attribution | No, for the released data | Official | Optional, and too large for a free session unless sampled |
| [FakeAVCeleb](https://sites.google.com/view/fakeavcelebdash-lab/download) | Not found | Audio-video deepfakes | Non-commercial research and education, no redistribution | **Yes** | Official for form terms | Optional |

WaveFake's English set comes from one speaker, so a split by speaker is impossible there. It stays out of training and is used to evaluate each vocoder.

## Text

| Dataset | Size | Generators or content | Licence | Request form | Evidence | Role |
|---|---|---|---|---|---|---|
| [MAGE](https://huggingface.co/datasets/yaful/MAGE) | 447,674 texts; train.csv 393 MB and test.csv 70 MB at an earlier commit | 27 generators, 14 domains | Apache 2.0 | No | Official | **Primary** training set, with a held-out generator |
| [HC3](https://huggingface.co/datasets/Hello-SimpleAI/HC3) | Nearly 40,000 questions with human and ChatGPT answers | ChatGPT across 5 domains | CC BY-SA 4.0, or stricter where a source is stricter; some splits are unknown | No | Official | **Primary** cross-dataset test |
| [M4GT-Bench](https://arxiv.org/pdf/2501.11012) | 86,782 human, 181,081 machine in one shared-task train split | 14 generators, 6 domains | CC BY-SA 4.0 for M4GT-Bench; original M4 not found | Not found | Secondary | Optional |
| [RAID](https://arxiv.org/abs/2405.07940) | Over 6 million generations | 11 models, 8 domains, 11 attacks | MIT in a shared-task table | No | Secondary | Adversarial evaluation, using a subset |
| [ChatGPT-Detector-Bias](https://huggingface.co/datasets/WxWx/ChatGPT-Detector-Bias) | 837 texts: 749 in `all_data.json`, plus 88 US essays that are only in the [authors' repository](https://github.com/Weixin-Liang/ChatGPT-Detector-Bias) | 91 TOEFL essays, 88 US essays, other human essays, GPT-3 and GPT-4 texts | MIT on the dataset card; the authors' repository has no LICENSE file at its root, and essays come from third parties | No | Official for files; secondary for licence | **Bias test only**, never for training |

The bias set is small. Two of the 91 TOEFL essays are duplicates, which leaves 89 distinct essays, so one extra false positive moves the non-native rate by about 1.1 points. The paper should report that rate with a Wilson confidence interval.

## Recommended core set

| Modality | Train and validate | In-dataset test | Unseen generator | Cross-dataset |
|---|---|---|---|---|
| Image | FaceForensics++ c23 and a GenImage subset | Same sources, disjoint identities | NeuralTextures, Midjourney, BigGAN | Celeb-DF v2 |
| Audio | ASVspoof 2019 LA train and dev | 2019 LA eval | A07–A18 except A16, which reuses A04; A19 also reuses A06 | ASVspoof 2021 DF and In-the-Wild |
| Text | MAGE, split by estimated prompt | Same source, disjoint prompts | The LLaMA family: 7B, 13B, 30B, 65B | HC3, excluding rows that open like MAGE training text |

Generators released after these datasets were built are absent from every test above. The paper must state that limit, and a small hand-collected set of recent generator outputs would be a useful extra test.

## Sources

- [FaceForensics++ repository](https://github.com/ondyari/faceforensics) and [Terms of Use](https://kaldir.vc.in.tum.de/faceforensics_tos.pdf)
- [Celeb-DF repository](https://github.com/yuezunli/celeb-deepfakeforensics) and [mirror listing size](https://huggingface.co/datasets/ej2/df/tree/main/data)
- [GenImage homepage](https://genimage-dataset.github.io) and [mirror](https://github.com/vtphatt2/GenImage-mirror)
- [ArtiFact paper](https://arxiv.org/pdf/2302.11970) and [review of AI-generated image detection](https://arxiv.org/pdf/2502.15176)
- [DFDC paper](https://arxiv.org/abs/2006.07397v1)
- [CIFAKE paper](https://arxiv.org/abs/2303.14126v1), [mirror](https://huggingface.co/datasets/yanbax/CIFAKE_autotrain_compatible) and [listing with a conflicting licence](https://www.emergentmind.com/topics/cifake-dataset)
- [ASVspoof 2019 record](https://datashare.ed.ac.uk/handle/10283/3336), [ASVspoof 2021 DF record](https://zenodo.org/record/4835108), [2022 licence announcement](https://auditory.org/mhonarc/2022/msg00573.html) and [survey with trial counts](https://arxiv.org/pdf/2404.13914)
- [WaveFake record](https://zenodo.org/record/5642694)
- [In-the-Wild page](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild)
- [ASVspoof 5 record](https://zenodo.org/records/14498691)
- [FakeAVCeleb request form](https://docs.google.com/forms/u/1/d/e/1FAIpQLSfPDd3oV0auqmmWEgCSaTEQ6CGpFeB-ozQJ35x-B_0Xjd93bw/viewform)
- [MAGE dataset card](https://huggingface.co/datasets/yaful/MAGE/raw/main/README.md)
- [HC3 dataset card](https://huggingface.co/datasets/Hello-SimpleAI/HC3/blob/main/README.md) and [per-split licence table](https://github.com/Hello-SimpleAI/chatgpt-comparison-detection)
- [GenAI Content Detection shared task](https://arxiv.org/pdf/2501.11012), source of the M4GT-Bench, RAID and MAGE table entries
- [Detector bias paper](https://arxiv.org/pdf/2304.02819) and [dataset mirror](https://huggingface.co/datasets/WxWx/ChatGPT-Detector-Bias/blob/main/README.md?code=true)
