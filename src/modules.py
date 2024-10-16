import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils import weight_norm, remove_weight_norm


def kaiserSincFilter(cutOff, halfWidth, kernelSize):
    halfSize = kernelSize // 2
    if kernelSize % 2 == 0:
        time = torch.arange(-halfSize, halfSize) + 0.5
    else:
        time = torch.arange(kernelSize) - halfSize

    attenuation = 2.285 * (halfSize - 1) * math.pi * 4 * halfWidth + 7.95

    if attenuation > 50.0:
        beta = 0.1102 * (attenuation - 8.7)
    elif attenuation >= 21.0:
        beta = 0.5842 * (attenuation - 21) ** 0.4 + 0.07886 * (attenuation - 21.0)
    else:
        beta = 0.0
    # Kaiser's estimation for beta
    window = torch.kaiser_window(kernelSize, False, beta)

    filter = (
        2 * cutOff * torch.sinc(2 * cutOff * time) * window
    )  # Kaiser Window with Sinc filter
    if cutOff != 0:
        filter /= filter.sum()
    return filter.view(1, 1, kernelSize)


class LowPassFilter(nn.Module):
    def __init__(self, kernelSize=6, cutOff=0.25, halfWidth=0.6):
        super().__init__()
        self.register_buffer("filter", kaiserSincFilter(cutOff, halfWidth, kernelSize))
        self.padLeft = kernelSize // 2 - int(kernelSize % 2 == 0)
        self.padRight = kernelSize // 2

    def forward(self, x):
        B, C, T = x.shape
        x = F.pad(x, (self.padLeft, self.padRight), "replicate")
        x = F.conv1d(x, self.filter.expand(C, -1, -1), groups=C)
        return x


class DownSampler(nn.Module):
    def __init__(self, ratio=2, kernelSize=12, cutOff=0.5, halfWidth=0.6):
        super().__init__()
        self.register_buffer(
            "filter", kaiserSincFilter(cutOff / ratio, halfWidth / ratio, kernelSize)
        )
        self.padLeft = kernelSize // 2 - int(kernelSize % 2 == 0)
        self.padRight = kernelSize // 2
        self.ratio = ratio

    def forward(self, x):
        B, C, T = x.shape
        x = F.pad(x, (self.padLeft, self.padRight), "replicate")  # Keep the shape
        x = F.conv1d(
            x, self.filter.expand(C, -1, -1), stride=self.ratio, groups=C
        )  # Channel-wise filtering
        return x


class UpSampler(nn.Module):
    def __init__(self, ratio=2, kernelSize=12, cutOff=0.5, halfWidth=0.6):
        super().__init__()
        self.register_buffer(
            "filter", kaiserSincFilter(cutOff / ratio, halfWidth / ratio, kernelSize)
        )
        self.pad = (
            kernelSize // ratio - 1
        )  # replicate padding to avoid transposed convolution out of sequence
        self.cropLeft = self.pad * ratio + (kernelSize - ratio) // 2
        self.cropRight = self.pad * ratio + (kernelSize - ratio + 1) // 2
        # Keep the sequence's length correct (L_out = L_in * ratio + kernelSize - ratio)
        self.ratio = ratio

    def forward(self, x):
        B, C, T = x.shape
        x = F.pad(x, (self.pad, self.pad), "replicate")
        x = self.ratio * F.conv_transpose1d(
            x, self.filter.expand(C, -1, -1), stride=self.ratio, groups=C
        )
        x = x[:, :, self.cropLeft : -self.cropRight]
        return x


class Snake(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        res = x.clone()
        res += 1.0 / (self.beta.exp() + 1e-8) * torch.sin(self.alpha.exp() * x).pow(2)
        return res


class AntiAliasingSnake(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.upSampler = UpSampler(ratio=2, kernelSize=12)
        self.downSampler = DownSampler(ratio=2, kernelSize=12)
        self.alpha = nn.Parameter(torch.zeros(1, channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        x = self.upSampler(x)
        res = x.clone()
        res += 1.0 / (self.beta.exp() + 1e-8) * torch.sin(self.alpha.exp() * x).pow(2)
        res = self.downSampler(res)
        return res


class Block(nn.Module):
    def __init__(self, channels, kernelSize=3, dilation=1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            channels, channels, kernelSize, dilation=dilation, padding="same"
        )
        self.act1 = Snake(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernelSize, padding="same")
        self.act2 = Snake(channels)

    def applyWeightNorm(self):
        self.conv1 = weight_norm(self.conv1)
        self.conv2 = weight_norm(self.conv2)

    def removeWeightNorm(self):
        self.conv1 = remove_weight_norm(self.conv1)
        self.conv2 = remove_weight_norm(self.conv2)

    def forward(self, x):
        res = x
        x = self.act1(x)
        x = self.conv1(x)
        x = self.act2(x)
        x = self.conv2(x)
        x += res
        return x


class ResLayer(nn.Module):
    def __init__(self, channels, kernelSize=(3, 5, 7), dilation=(1, 3, 5)):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.kernelSize = kernelSize
        self.dilation = dilation
        for i in range(len(kernelSize)):
            for j in range(len(dilation)):
                self.blocks.append(Block(channels, kernelSize[i], dilation[j]))

    def applyWeightNorm(self):
        for i in range(len(self.blocks)):
            self.blocks[i].applyWeightNorm()

    def removeWeightNorm(self):
        for i in range(len(self.blocks)):
            self.blocks[i].removeWeightNorm()

    def forwardOneKernel(self, x, kernelID):
        out = self.blocks[kernelID * len(self.dilation)](x)
        for i in range(1, len(self.dilation)):
            out = self.blocks[kernelID * len(self.dilation) + i](out)
        return out

    def forward(self, x):
        sum = self.forwardOneKernel(x, 0)
        for i in range(1, len(self.kernelSize)):
            sum += self.forwardOneKernel(x, i)
        sum /= len(self.kernelSize)
        return sum
