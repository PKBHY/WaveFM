# WaveFM

WaveFM: A High-Fidelity and Efficient Vocoder Based on Flow Matching. See audio demo at https://PKBHY.github.io/WaveFM **(fixed on 24 Nov, 2024, GMT)**

The original and distilled checkpoints are provided in `src`, which are currently for inference purposes only. 

## Basic Usage

First modify the paths for `training`, `distillation` and `inference` in `params.py` as needed. 

```
cd src

# Extracting Mel-spectrograms for training and inference process
python dataset.py -i your_audio_path -o your_mel_saving_path

# Training
python train.py

# Distillation
python distillation.py

# Inference
python inference.py
```

## Package Requirements

WaveFM has been tested on `python 3.9` with the following requirements, and it should also work fine with the latest version.

```
torch           2.0.1+cu118
torchaudio      2.0.2+cu118
tqdm            4.66.4
```
