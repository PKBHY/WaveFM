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
import torch.optim.lr_scheduler as lrs

from params import params
from dataset import AudioMelSet
from models import Velocity, getSTFTLoss
from params import params
from torchdiffeq import odeint


def distillation(mixTraining=params["distillWithHybridPrecision"]):
    
    trainData = AudioMelSet(params["distillAudiosPath"], params["distillMelsPath"])
    trainLoader = torch.utils.data.DataLoader(
        trainData,
        batch_size=params["distillBatch"],
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )

    device = params["distillDevice"]
    betas = params["distillBetas"]
    weightDecay = params["distillWeightDecay"]

    velocity = Velocity().to(device)
    velocityTarget = Velocity().to(device)
    velocityAnswer = Velocity().to(device)

    vOptimizer = optim.AdamW(
        velocity.parameters(),
        lr=params["distillInitialLR"],
        betas=betas,
        weight_decay=weightDecay,
    )

    vScheduler = lrs.CosineAnnealingLR(vOptimizer, T_max=params['distillSteps'] , eta_min=params['distillFinalLR'])
        
    if os.path.exists(params["distillModelPath"]):
        all = torch.load(params["distillModelPath"])
        velocityAnswer.load_state_dict(all["velocity"])
    else:
        raise Exception("Your model path to be distilled doesn't exist.")

    if os.path.exists(params["distillCheckPointPath"]):

        all = torch.load(params["distillCheckPointPath"])
        velocity.load_state_dict(all["velocity"])
        vOptimizer.load_state_dict(all["vOptimizer"])
        velocityTarget.load_state_dict(all["velocityTarget"])
        vScheduler.load_state_dict(all["vScheduler"])

        deltaT = all["deltaT"]
        nowStep = all["step"]
        nowEpoch = all["epoch"]

        for param_group in vOptimizer.param_groups:
            param_group["betas"] = betas

        for param_group in vOptimizer.param_groups:
            param_group["weight_decay"] = weightDecay

    else:
        nowStep = 0
        nowEpoch = 0
        deltaT = params["distillDeltaT"]
        all = torch.load(params["distillModelPath"])
        velocity.load_state_dict(all["velocity"], strict=True)
        velocityTarget.load_state_dict(all["velocity"], strict=True)

        path = params["distillCheckPointPath"]

        pos = path.rfind("_")
        if pos == -1 or pos == len(path) - 1 or not path[pos + 1:].isdigit():
            path = path + "_" + str(nowStep)
        else:
            path = path[:pos] + "_" + str(nowStep)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "velocity": velocity.state_dict(),
                "vOptimizer": vOptimizer.state_dict(),
                "velocityTarget": velocityTarget.state_dict(),
                "vScheduler": vScheduler.state_dict(),
                "step": nowStep,
                "epoch": nowEpoch,
                "distilled": True,
                "deltaT": deltaT,
            },
            path,
        )

    scaler = torch.cuda.amp.GradScaler(enabled=mixTraining)

    melProcessor = torchaudio.transforms.MelSpectrogram(
        sample_rate=params["sampleRate"],
        n_fft=params["fftSize"],
        win_length=params["windowSize"],
        hop_length=params["hopSize"],
        n_mels=params["melBands"],
        f_min=params["fmin"],
        f_max=params["fmax"],
        center=True,
        power=2,
        pad_mode="reflect",
    ).to(device)

    maximumEnergy = torch.sqrt(torch.tensor(params["melBands"] * 32768.0))
    velocity.train()
    velocityTarget.eval()
    velocityAnswer.eval()
    meanMelLoss = None
    meanSTFTLoss = None
    meanVelocityMSELoss = None
    while True:

        tqdmLoader = tqdm(
            trainLoader, desc=f"distill Epoch: {nowEpoch}, starting step={nowStep}"
        )
        for audios, mels in tqdmLoader:

            with torch.cuda.amp.autocast(enabled=mixTraining):

                x1 = audios.to(device)
                mels = mels.to(device)
                energy = mels.exp().sum(dim=1).sqrt().unsqueeze(1)
                sigma = F.interpolate(
                    (energy / maximumEnergy).clamp(min=0.001),
                    size=(energy.size(-1) * params["hopSize"]),
                )
                x0 = sigma * torch.randn_like(sigma)

                t = torch.nn.init.trunc_normal_(torch.zeros(x1.size(0),1,1,device=x1.device),mean=0,std=0.33,a=0,b=0.99)
               
                xt = t * x1 + (1 - t) * x0
                tPrime = (t + deltaT).clamp(max=1.0)
             
                with torch.no_grad():
                    # Producing Target With Euler Solver
                    heunVelocity1 = (velocityAnswer(xt, mels, t) - xt) / (1 - t)
                    xtPrime = xt + heunVelocity1 * (tPrime - t)

                    marginCondition = ((t + deltaT) >= 0.99).bool()
                    predict = velocityTarget(xtPrime, mels, tPrime)
                    target = torch.where(marginCondition, x1, predict).detach()

                # Producing Target With Heun Solver
                # with torch.no_grad():
                #     heunVelocity1 = (velocityAnswer(xt, mels, t) - xt) / (1 - t)
                #     xtPrime = xt + heunVelocity1 * (tPrime - t)
                #     heunVelocity2 = (
                #                          velocityAnswer(xtPrime, mels, tPrime) - xtPrime
                #                     ) / (1 - tPrime)
                #     teacherPredict = xt + (heunVelocity1 + heunVelocity2) / 2.0 * (
                #             tPrime - t
                #     )
                #     marginCondition = ((t + deltaT) >= 0.99).bool()
                #     predict = velocityTarget(teacherPredict, mels, tPrime)
                #     target = torch.where(marginCondition, x1, predict).detach()

                studentPredict = velocity(xt, mels, t)
                delta = studentPredict - target
                fakeMels = melProcessor(studentPredict).clamp(min=1e-5).log()
                realMels = melProcessor(target).clamp(min=1e-5).log()

                melLoss = ((fakeMels - realMels).abs()).mean()
                STFTLoss = getSTFTLoss(target, studentPredict)
                velocityMSELoss = (delta.pow(2) * (1.0 / (1 - t).clamp(min=0.1))).mean()
                loss = (
                        velocityMSELoss
                        + 0.02 * melLoss
                        + 0.02 * STFTLoss
                )

                vOptimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(vOptimizer)
                scaler.step(vOptimizer)
                scaler.update()
                vScheduler.step()
                
                with torch.no_grad():
                    # EMA parameters update
                    velocityDict = dict(velocity.named_parameters())
                    velocityTargetDict = dict(velocityTarget.named_parameters())

                    for name in velocityTargetDict:
                        velocityTargetDict[name].mul_(0.999).add_(
                            0.001 * velocityDict[name]
                        )

                if meanMelLoss is None:
                    meanMelLoss = melLoss.item()
                    meanVelocityMSELoss = velocityMSELoss.sqrt().item()
                    meanSTFTLoss = STFTLoss.sqrt().item()
                else:
                    meanMelLoss = meanMelLoss * 0.99 + 0.01 * melLoss.item()
                    meanVelocityMSELoss = meanVelocityMSELoss * 0.99 + 0.01 * velocityMSELoss.sqrt().item()

                    meanSTFTLoss = meanSTFTLoss * 0.99 + 0.01 * STFTLoss.sqrt().item()

                tqdmLoader.set_postfix(
                    MSELoss=round(meanVelocityMSELoss, 4),
                    MelLoss=round(meanMelLoss, 4),
                    STFTLoss=round(meanSTFTLoss, 4),
                    dt=round(deltaT, 4),
                    LR=f"{vScheduler.get_last_lr()[0]:.2e}"
                )

                nowStep += 1

                if nowStep % params["distillCheckPointSavingStep"] == 0:

                    path = params["distillCheckPointPath"]
                    pos = path.rfind("_")
                    if (
                        pos == -1
                        or pos == len(path) - 1
                        or not path[pos + 1:].isdigit()
                    ):
                        path = path + "_" + str(nowStep)
                    else:
                        path = path[:pos] + "_" + str(nowStep)
                        
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    torch.save(
                        {
                            "velocity": velocity.state_dict(),
                            "velocityTarget": velocityTarget.state_dict(),
                            "vScheduler": vScheduler.state_dict(),
                            "vOptimizer": vOptimizer.state_dict(),
                            "step": nowStep,
                            "epoch": nowEpoch,
                            "distilled": True,
                            "deltaT": deltaT,
                        },
                        path,
                    )
                
                # if nowStep % params["distillLearnRateDecayStep"] == 0:
                #     for param_group in vOptimizer.param_groups:
                #         param_group["lr"] *= gamma

                if nowStep >= params["distillSteps"]:
                    return

        nowEpoch += 1


if __name__ == "__main__":
    distillation()
