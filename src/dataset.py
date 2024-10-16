import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm
from params import params
from torch.utils.data.dataloader import Dataset, DataLoader
import os
import math
import argparse


def audio2Mel(audioPath, melSavingPath, device="cuda:0"):
    
    melProcessor = torchaudio.transforms.MelSpectrogram(
        sample_rate=params["sampleRate"],
        n_fft=params["fftSize"],
        win_length=params["windowSize"],
        hop_length=params["hopSize"],
        n_mels=params["melBands"],
        f_min=params["fmin"],
        f_max=params["fmax"],
        power=2,
        center=True,
        pad_mode="reflect",
    ).to(device)

    files = [name for name in os.listdir(audioPath) if name.endswith(".wav")]
    totalTime = 0.0

    for audioName in tqdm(files, desc="Transforming audios"):
        waveform, sampleRate = torchaudio.load(audioPath + "/" + audioName)

        melSpectrogram = melProcessor(waveform.to(device))
        melSpectrogram = melSpectrogram.clamp(min=1e-5).log().squeeze(0)
        os.makedirs(melSavingPath, exist_ok=True)
        torch.save(melSpectrogram.cpu(), melSavingPath + "/" + audioName[:-4] + ".mel")
        totalTime += waveform.size(-1)

    print("Audios' total length : {} seconds".format(totalTime / params["sampleRate"]))


class AudioMelSet(Dataset):

    def __init__(self, audioPath, melPath):
        self.sampleRate = params["sampleRate"]
        self.audioPath = audioPath
        self.melPath = melPath
        self.nameList = [name[:-4] for name in os.listdir(melPath)]

    def __getitem__(self, index):
        waveform, sampleRate = torchaudio.load(
            self.audioPath + "/" + self.nameList[index] + ".wav"
        )
        melSpectrogram = torch.load(self.melPath + "/" + self.nameList[index] + ".mel")
        melLen = params["melTrainWindow"]
        audioLen = melLen * params["hopSize"]

        if melSpectrogram.size(-1) < melLen:

            melSpectrogram = F.pad(
                melSpectrogram,
                (0, melLen - melSpectrogram.size(-1)),
                mode="constant",
                value=math.log(1e-5),
            )
            waveform = F.pad(
                waveform, (0, audioLen - waveform.size(-1)), mode="constant", value=0
            )
            return waveform, melSpectrogram
        else:
            start = torch.randint(
                low=0, high=melSpectrogram.size(-1) - melLen + 1, size=(1,)
            ).item()
            end = start + melLen
            melWindow = melSpectrogram[..., start:end]
            if waveform.size(-1) < end * params["hopSize"]:
                waveform = F.pad(
                    waveform,
                    (0, end * params["hopSize"] - waveform.size(-1)),
                    mode="constant",
                    value=0,
                )

            audioWindow = waveform[
                ..., start * params["hopSize"] : end * params["hopSize"]
            ]
            return audioWindow, melWindow

    def __len__(self):
        return len(self.nameList)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transform audios into mel-spectrograms."
    )
    parser.add_argument(
        "-i",
        "--audioPath",
        type=str,
        default="./LibriTTS/train",
        help="Audio file path.",
    )
    parser.add_argument(
        "-o",
        "--melSavingPath",
        type=str,
        default="./trainMel",
        help="Mel-spectrogram saving path.",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda:0",
        help="Processing device (cpu/cuda:0)",
    )
    args = parser.parse_args()

    audio2Mel(args.audioPath, args.melSavingPath, args.device)
