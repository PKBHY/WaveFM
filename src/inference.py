import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.functional as F
import torchaudio
import os
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import numpy as np

from params import params
from dataset import AudioMelSet
from models import Velocity

import time


def inference():
    
    with torch.no_grad():
        
        device = params["inferenceDevice"]
        sampleRate = params["sampleRate"]

        velocity = Velocity().to(device)
        distilled = False
        if os.path.exists(params["inferenceCheckPointPath"]):

            all = torch.load(params["inferenceCheckPointPath"])
            velocity.load_state_dict(all["velocity"], strict=True)

            nowStep = all["step"]
            print(f"{nowStep} steps model is loaded.")
            print(
                f"Params: {sum([param.numel() for param in velocity.parameters()])/1e6}M"
            )
            if all.get("distilled") is not None:
                distilled = True
                print(f"The model is distilled.")
            else:
                print("The model is not distilled.")

        else:
            raise Exception("Your checkpoint path doesn't exist.")

        maximumEnergy = torch.sqrt(torch.tensor(params["melBands"] * 32768.0))
        melPath = params["inferenceMelsPath"]
        savingPath = params["inferenceSavingPath"]
        allFiles = os.listdir(melPath)
        files = [name for name in allFiles if name.endswith(".mel")]
        loader = tqdm(files, desc="Inference ")
        inferenceTime = 0
        audioTime = 0
        NFE = 0
        
        amount = 0
        velocity.eval()
    
        for name in loader:
            amount += 1
            melSpectrogram = torch.load(melPath + "/" + name).unsqueeze(0).to(device)
            
            start = time.perf_counter() 
            energy = melSpectrogram.exp().sum(dim=1).sqrt().unsqueeze(1)
            sigma = F.interpolate(
                (energy / maximumEnergy).clamp(min=0.001),
                size=(energy.size(-1) * params["hopSize"]),
            )
            x0 = 1.0 * sigma * torch.randn_like(sigma)

            if distilled:
                predict = velocity(x0, melSpectrogram, torch.zeros(1, 1).to(device))
                NFE += 1
                end = time.perf_counter()

            else:
                
                deltaT = 1.0 / params['inferenceSteps']
                tNow = torch.zeros(1, 1).to(device)
                
                for _ in range(params['inferenceSteps'] - 1):
                 
                    x0 = x0 * (1 - ( tNow + deltaT )) / (1 - tNow) \
                         + velocity(x0, melSpectrogram, tNow) * ( deltaT / (1 - tNow))
                         
                    tNow += deltaT
                    
                predict = velocity(x0 , melSpectrogram, tNow)
                NFE += params['inferenceSteps']
                end = time.perf_counter()
                
            inferenceTime += end - start
            audioTime += melSpectrogram.size(-1) * 256.0 / sampleRate
            os.makedirs(savingPath, exist_ok=True)
            torchaudio.save(
                savingPath + "/" + name[:-4] + ".wav", predict[0].cpu(), sampleRate
            )
            
            loader.set_postfix(
                NFE=round(NFE / amount, 2),
                AudioTime=round(audioTime, 2),
                InferenceTime=round(inferenceTime, 2),
                RTF=round(audioTime / inferenceTime, 2),
            )

if __name__ == "__main__":
    inference()
