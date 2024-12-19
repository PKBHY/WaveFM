import torch
import torch.optim as optim
import torch.nn.functional as F
import torchaudio
import torch.optim.lr_scheduler as lrs
import os
from tqdm import tqdm

from dataset import AudioMelSet
from models import Velocity, getSTFTLoss
from params import params


def train(mixTraining = params["trainWithHybridPrecision"]):
    trainData = AudioMelSet(params["trainAudiosPath"], params["trainMelsPath"])
    trainLoader = torch.utils.data.DataLoader(
        trainData,
        batch_size=params["trainBatch"],
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )

    device = params["trainDevice"]
    betas = params["trainBetas"]
    weightDecay = params["trainWeightDecay"]

    velocity = Velocity().to(device)

    vOptimizer = optim.AdamW(
        velocity.parameters(),
        lr=params["trainInitialLR"],
        betas=betas,
        weight_decay=weightDecay,
    )

    vScheduler = lrs.CosineAnnealingLR(vOptimizer, T_max=params['trainSteps'] , eta_min=params['trainFinalLR'])
    
    if os.path.exists(params["trainCheckPointPath"]):
        all = torch.load(params["trainCheckPointPath"])
        velocity.load_state_dict(all["velocity"], strict=False)
        vOptimizer.load_state_dict(all["vOptimizer"])
        vScheduler.load_state_dict(all["vScheduler"])

        nowStep = all["step"]
        nowEpoch = all["epoch"]

        for param_group in vOptimizer.param_groups:
            param_group["weight_decay"] = weightDecay
        for param_group in vOptimizer.param_groups:
            param_group["betas"] = betas

    else:
        nowStep = 0
        nowEpoch = 0

        path = params["trainCheckPointPath"]
        for para in velocity.parameters():
            para.data.clamp_(-0.1, 0.1)

        pos = path.rfind("_")
        if pos == -1 or pos == len(path) - 1 or not path[pos + 1 :].isdigit():
            path = path + "_" + str(nowStep)
        else:
            path = path[:pos] + "_" + str(nowStep)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "velocity": velocity.state_dict(),
                "vOptimizer": vOptimizer.state_dict(),
                "vScheduler": vScheduler.state_dict(),
                "step": nowStep,
                "epoch": nowEpoch,
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
    meanMelLoss = None
    meanVelocityMSELoss = None
    meanSTFTLoss = None
    
    velocity.train()
    vScheduler.eta_min=params['trainFinalLR']
    
    while True:
        tqdmLoader = tqdm(
            trainLoader, desc=f"train Epoch: {nowEpoch}, starting step={nowStep}"
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
                epsilon = torch.randn_like(sigma)
                x0 = sigma * epsilon

                t = torch.rand(x0.size(0), 1, 1).to(device)
                xt = x0 * (1 - t) + x1 * t

                predict = velocity(xt, mels, t)
                delta = predict - x1
                fakeMels = melProcessor(predict).clamp(min=1e-5).log()
                realMels = melProcessor(x1).clamp(min=1e-5).log()
                scale = 1.0 / (1 - t).clamp(min=0.1)
                melLoss = ((fakeMels - realMels).abs()).mean()

                STFTLoss = getSTFTLoss(x1, predict)
                velocityMSELoss = (delta.pow(2) * scale).mean()

                loss = (
                    velocityMSELoss
                    + 0.02 * melLoss
                    + 0.02 * STFTLoss
                )

                if meanMelLoss is None:
                    meanMelLoss = melLoss.item()
                    meanVelocityMSELoss = velocityMSELoss.sqrt().item()
                    meanSTFTLoss = STFTLoss.item()

                else:
                    meanMelLoss = meanMelLoss * 0.99 + 0.01 * melLoss.item()
                    meanVelocityMSELoss = (
                        meanVelocityMSELoss * 0.99
                        + 0.01 * velocityMSELoss.sqrt().item()
                    )
                    meanSTFTLoss = meanSTFTLoss * 0.99 + 0.01 * STFTLoss.item()

                tqdmLoader.set_postfix(
                    MSELoss=round(meanVelocityMSELoss, 4),
                    MelLoss=round(meanMelLoss, 4),
                    STFTLoss=round(meanSTFTLoss, 4),
                    LR=f"{vScheduler.get_last_lr()[0]:.2e}"
                )

                vOptimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(vOptimizer)
                scaler.step(vOptimizer)
                scaler.update()
                vScheduler.step()
                nowStep += 1

                if nowStep % params["trainCheckPointSavingStep"] == 0:

                    path = params["trainCheckPointPath"]
                    pos = path.rfind("_")
                    if (
                        pos == -1
                        or pos == len(path) - 1
                        or not path[pos + 1 :].isdigit()
                    ):
                        path = path + "_" + str(nowStep)
                    else:
                        path = path[:pos] + "_" + str(nowStep)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    torch.save(
                        {
                            "velocity": velocity.state_dict(),
                            "vOptimizer": vOptimizer.state_dict(),
                            "vScheduler": vScheduler.state_dict(),
                            "step": nowStep,
                            "epoch": nowEpoch,
                        },
                        path,
                    )
                    

                if nowStep >= params["trainSteps"]:
                    return

        nowEpoch += 1


if __name__ == "__main__":
    train()
