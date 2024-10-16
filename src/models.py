import torch
import torch.nn as nn
import torch.nn.functional as F
from params import params
from modules import AntiAliasingSnake, ResLayer, Snake, UpSampler, DownSampler
from torch.nn.utils import weight_norm, remove_weight_norm


def getSTFTLoss(
    answer,
    predict,
    fft_sizes=(1024, 2048, 512),
    hop_sizes=(128, 256, 64),
    win_lengths=(512, 1024, 256),
    window=torch.hann_window,
):
    
    loss = 0
    for i in range(len(fft_sizes)):
        
        answerStft = torch.view_as_real(
            torch.stft(
                answer.squeeze(1),
                n_fft=fft_sizes[i],
                hop_length=hop_sizes[i],
                win_length=win_lengths[i],
                window=window(win_lengths[i], device=answer.device),
                return_complex=True,
            )
        )
        predictStft = torch.view_as_real(
            torch.stft(
                predict.squeeze(1),
                n_fft=fft_sizes[i],
                hop_length=hop_sizes[i],
                win_length=win_lengths[i],
                window=window(win_lengths[i], device=predict.device),
                return_complex=True,
            )
        )

        answerStftMag = answerStft[..., 0] ** 2 + answerStft[..., 1] ** 2

        predictStftMag = predictStft[..., 0] ** 2 + predictStft[..., 1] ** 2
        
        magnitude_threshold = 1e-6
        mask = (answerStftMag > magnitude_threshold) & (predictStftMag > magnitude_threshold)
        
        answerStftMag = torch.sqrt(answerStftMag + magnitude_threshold)
        predictStftMag = torch.sqrt(predictStftMag + magnitude_threshold)
        
        answerStftPha = torch.atan2(
            answerStft[..., 1][mask], answerStft[..., 0][mask]
        )
        predictStftPha = torch.atan2(
            predictStft[...,1][mask],predictStft[..., 0][mask]
        )
        
        deltaPhase = answerStftPha - predictStftPha
        
        loss +=  (torch.atan2(torch.sin(deltaPhase), torch.cos(deltaPhase)).abs()).mean()
        
        loss += (answerStftMag.log() - predictStftMag.log()).abs().mean()
        
    return loss / len(fft_sizes)


class Velocity(nn.Module):

    @staticmethod
    def timeEmbedding(t):
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)  # batch -> batch*1
        if len(t.shape) == 3:
            t = t.squeeze(-1)  # batch*1*1 -> batch*1

        pos = torch.arange(64, device=t.device).unsqueeze(0)  # 1*64
        table = 100 * t * 10.0 ** (pos * 4.0 / 63.0)  # batch*64

        return torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # batch*128

    def __init__(
        self,
        channels=params["velocityChannels"],
        upSampleRates=params["velocityUpSampleRates"],
        kernelSizesUp=params["velocityKernelSizesUp"],
        dilationsUp=params["velocityDilationsUp"],
        kernelSizesDown=params["velocityKernelSizesDown"],
        dilationsDown=params["velocityDilationsDown"],
    ):
        super().__init__()

        self.timePre0 = nn.Linear(128, params["timeEmbeddingSize"])
        self.timePre1 = nn.Linear(
            params["timeEmbeddingSize"], params["timeEmbeddingSize"]
        )
        self.SiLU = nn.SiLU()
        self.upSampleRates = upSampleRates

        size = 7
        self.convUpIn = nn.Conv1d(
            params["melBands"], channels[0], size, 1, padding="same"
        )
        self.convDownIn = nn.Conv1d(1, channels[-1], size, padding="same")

        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        for i in range(len(upSampleRates)):

            self.ups.append(
                
                nn.ConvTranspose1d(
                    channels[i],
                    channels[i + 1],
                    kernel_size=2 * upSampleRates[i],
                    stride=upSampleRates[i],
                    padding=upSampleRates[i] // 2,
                ),

            )  # stride=2kernel=4padding


            self.downs.append(
                
                nn.Conv1d(
                    channels[i + 1],
                    channels[i],
                    kernel_size= 2 * upSampleRates[i] + 1,
                    stride=upSampleRates[i],
                    padding=upSampleRates[i],
                )
           
            )

        self.resLayerUps = nn.ModuleList()
        self.resLayerDowns = nn.ModuleList()
        self.timeDowns = nn.ModuleList()

        for i in range(len(upSampleRates)):
            
            self.timeDowns.append(
                nn.Linear(params["timeEmbeddingSize"], channels[i + 1])
            )
            
            self.resLayerUps.append(
                ResLayer(channels[i + 1], kernelSizesUp[i], dilationsUp[i])
            )
            
            self.resLayerDowns.append(
               ResLayer(channels[i + 1], kernelSizesDown[i], dilationsDown[i])
            )

        self.convUpOut = nn.Conv1d(channels[-1], 1, size, 1, padding="same")
        self.actUpOut = Snake(channels=channels[-1])

    def applyWeightNorm(self):
        self.convDownIn = weight_norm(self.convDownIn)
        self.convUpIn = weight_norm(self.convUpIn)
        self.convUpOut = weight_norm(self.convUpOut)

        for i in range(len(self.resLayerUps)):
            self.resLayerUps[i].applyWeightNorm()
        for i in range(len(self.resLayerDowns)):
            self.resLayerDowns[i].applyWeightNorm()
        for i in range(len(self.ups)):
            self.ups[i] = weight_norm(self.ups[i])
        for i in range(len(self.downs)):
            self.downs[i] = weight_norm(self.downs[i])

    def removeWeightNorm(self):
        self.convDownIn = remove_weight_norm(self.convDownIn)
        self.convUpIn = remove_weight_norm(self.convUpIn)
        self.convUpOut = remove_weight_norm(self.convUpOut)

        for i in range(len(self.resLayerUps)):
            self.resLayerUps[i].removeWeightNorm()
        for i in range(len(self.resLayerDowns)):
            self.resLayerDowns[i].removeWeightNorm()
        for i in range(len(self.ups)):
            self.ups[i] = remove_weight_norm(self.ups[i])
        for i in range(len(self.downs)):
            self.downs[i] = remove_weight_norm(self.downs[i])

    def forward(self, x, melSpectrogram, t):
        
        timeEmbedding = self.timeEmbedding(t)
        timeEmbedding = self.SiLU(self.timePre0(timeEmbedding))
        timeEmbedding = self.SiLU(self.timePre1(timeEmbedding))

        x = self.convDownIn(x)

        skipConnections = [x.clone()]
        for i in range(len(self.downs) - 1, -1, -1):
            
            x += self.timeDowns[i](timeEmbedding).unsqueeze(-1)
            x = self.resLayerDowns[i](x)
            x = self.downs[i](x)

            skipConnections.append(x.clone())

        melSpectrogram = self.convUpIn(melSpectrogram) 
        melSpectrogram += skipConnections[-1]

        for i in range(len(self.ups)):
          
            melSpectrogram = self.ups[i](melSpectrogram)
            melSpectrogram += skipConnections[-i - 2]
            melSpectrogram = self.resLayerUps[i](melSpectrogram)

        out = self.actUpOut(melSpectrogram)
        out = self.convUpOut(out)
        out = torch.tanh(out)

        return out
