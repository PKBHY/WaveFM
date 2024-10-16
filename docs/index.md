---
layout: default
---

Flow matching offers a robust and stable approach to training diffusion models. However, directly applying flow matching to neural vocoders can result in subpar audio quality. In this work, we present WaveFM, a reparameterized flow matching model for mel-spectrogram conditioned speech synthesis, designed to enhance both sample quality and generation speed for diffusion vocoders. Since mel-spectrograms represent the energy distribution of waveforms, WaveFM adopts a mel-conditioned prior distribution instead of a standard Gaussian prior to minimize unnecessary transportation costs during synthesis. Moreover, while most diffusion vocoders rely on a single loss function, we argue that incorporating auxiliary losses, including a refined multi-resolution STFT loss, can further improve audio quality. To speed up inference without degrading sample quality significantly, we introduce a tailored consistency distillation method for WaveFM. Experimental results demonstrate that our model achieves superior performance in both quality and efficiency compared to previous diffusion vocoders, while enabling waveform generation in a single inference step while enabling waveform generation in a single inference step.

# Model

![Model](./model.png)

The total amount of parameters is `19.5M`. The 128-dim time embedding is expanded to 512-dim after two linear-SiLU layers, and is then reshaped to the desired shape of each resolution. `Conv1d` and `ConvTranspose1d` are set with parameters `(output channel, kernel width, dilation, padding)`. In ResBlock `Conv1d` takes same padding. Each ResLayer is defined with a kernel list and a dilation list, and their cross-product of define the shape of the ResBlock matrix and the convolutional layers of each ResBlock. On the left column are downsampling ResLayers, each containing a `4 x 1` ResBlock matrix, while on the right columns are upsampling ResLayers, each containing a `3 x 3` ResBlock matrix, following the structure from HifiGAN. In each ResBlock the number of channels is unchanged.

For detailed parameter settings please refer to `WaveFM/src/params.py`.

# Audio Samples

<table>
  <thead>
    <tr>
      <td align="center"><b>Groundtruth</b><br>
      </td>
      <td align="center"><b>WaveFM (6 step)</b><br>
      </td>
      <td align="center"><b>WaveFM (1 step)</b><br>
      </td>
      <td align="center"><b>Hifi-GAN</b><br>
      </td>
      <td align="center"><b>Diffwave</b><br>
      </td>
      <td align="center"><b>PriorGrad</b><br>
      </td>
      <td align="center"><b>FreGrad</b><br>
      </td>
      <td align="center"><b>FastDiff</b><br>
      </td>
    </tr>
  </thead>
  <tbody>
    <tr><td colspan="8">Male 1</td></tr>
  </tbody>
  <tbody>
    <tr>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/GT/1089_134686_000007_000005.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM1/1089_134686_000007_000005.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM6/1089_134686_000007_000005.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/HifiGAN/1089_134686_000007_000005.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/DiffWave/1089_134686_000007_000005.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/PriorGrad/1089_134686_000007_000005.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FreGrad/1089_134686_000007_000005.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FastDiff/1089_134686_000007_000005.wav"></audio>
      </td>
    </tr>
  </tbody>
  <tbody>
    <tr><td colspan="8">Male 2</td></tr>
  </tbody>
  <tbody>
    <tr>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/GT/1089_134686_000024_000007.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM1/1089_134686_000024_000007.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM6/1089_134686_000024_000007.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/HifiGAN/1089_134686_000024_000007.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/DiffWave/1089_134686_000024_000007.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/PriorGrad/1089_134686_000024_000007.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FreGrad/1089_134686_000024_000007.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FastDiff/1089_134686_000024_000007.wav"></audio>
      </td>
    </tr>
  </tbody>
  <tbody>
    <tr><td colspan="8">Male 3</td></tr>
  </tbody>
  <tbody>
    <tr>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/GT/1188_133604_000011_000003.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM1/1188_133604_000011_000003.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM6/1188_133604_000011_000003.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/HifiGAN/1188_133604_000011_000003.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/DiffWave/1188_133604_000011_000003.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/PriorGrad/1188_133604_000011_000003.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FreGrad/1188_133604_000011_000003.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FastDiff/1188_133604_000011_000003.wav"></audio>
      </td>
    </tr>
  </tbody>
  <tbody>
    <tr><td colspan="8">Male 4</td></tr>
  </tbody>
  <tbody>
    <tr>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/GT/1188_133604_000018_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM1/1188_133604_000018_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM6/1188_133604_000018_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/HifiGAN/1188_133604_000018_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/DiffWave/1188_133604_000018_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/PriorGrad/1188_133604_000018_000000.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FreGrad/1188_133604_000018_000000.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FastDiff/1188_133604_000018_000000.wav"></audio>
      </td>
    </tr>
  </tbody>
  <tbody>
    <tr><td colspan="8">Male 5</td></tr>
  </tbody>
  <tbody>
    <tr>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/GT/1320_122612_000013_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM1/1320_122612_000013_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM6/1320_122612_000013_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/HifiGAN/1320_122612_000013_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/DiffWave/1320_122612_000013_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/PriorGrad/1320_122612_000013_000000.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FreGrad/1320_122612_000013_000000.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FastDiff/1320_122612_000013_000000.wav"></audio>
      </td>
    </tr>
  </tbody>
  <tbody>
    <tr><td colspan="8">Female 1</td></tr>
  </tbody>
  <tbody>
    <tr>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/GT/121_127105_000014_000001.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM1/121_127105_000014_000001.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM6/121_127105_000014_000001.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/HifiGAN/121_127105_000014_000001.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/DiffWave/121_127105_000014_000001.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/PriorGrad/121_127105_000014_000001.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FreGrad/121_127105_000014_000001.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FastDiff/121_127105_000014_000001.wav"></audio>
      </td>
    </tr>
  </tbody>
  <tbody>
    <tr><td colspan="8">Female 2</td></tr>
  </tbody>
  <tbody>
    <tr>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/GT/121_127105_000040_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM1/121_127105_000040_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM6/121_127105_000040_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/HifiGAN/121_127105_000040_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/DiffWave/121_127105_000040_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/PriorGrad/121_127105_000040_000000.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FreGrad/121_127105_000040_000000.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FastDiff/121_127105_000040_000000.wav"></audio>
      </td>
    </tr>
  </tbody>
  <tbody>
    <tr><td colspan="8">Female 3</td></tr>
  </tbody>
  <tbody>
    <tr>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/GT/237_126133_000033_000001.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM1/237_126133_000033_000001.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM6/237_126133_000033_000001.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/HifiGAN/237_126133_000033_000001.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/DiffWave/237_126133_000033_000001.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/PriorGrad/237_126133_000033_000001.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FreGrad/237_126133_000033_000001.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FastDiff/237_126133_000033_000001.wav"></audio>
      </td>
    </tr>
  </tbody>
  <tbody>
    <tr><td colspan="8">Female 4</td></tr>
  </tbody>
  <tbody>
    <tr>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/GT/237_134493_000003_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM1/237_134493_000003_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM6/237_134493_000003_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/HifiGAN/237_134493_000003_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/DiffWave/237_134493_000003_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/PriorGrad/237_134493_000003_000000.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FreGrad/237_134493_000003_000000.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FastDiff/237_134493_000003_000000.wav"></audio>
      </td>
    </tr>
  </tbody>
  <tbody>
    <tr><td colspan="8">Female 5</td></tr>
  </tbody>
  <tbody>
    <tr>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/GT/1284_1181_000045_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM1/1284_1181_000045_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM6/1284_1181_000045_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/HifiGAN/1284_1181_000045_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/DiffWave/1284_1181_000045_000000.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/PriorGrad/1284_1181_000045_000000.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FreGrad/1284_1181_000045_000000.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FastDiff/1284_1181_000045_000000.wav"></audio>
      </td>
    </tr>
  </tbody>
  <tbody>
    <tr><td colspan="8">Music 1 (out of distribution)</td></tr>
  </tbody>
  <tbody>
    <tr>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/GT/Al%20James%20-%20Schoolboy%20Facination.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM1/Al%20James%20-%20Schoolboy%20Facination.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM6/Al%20James%20-%20Schoolboy%20Facination.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/HifiGAN/Al%20James%20-%20Schoolboy%20Facination.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/DiffWave/Al%20James%20-%20Schoolboy%20Facination.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/PriorGrad/Al%20James%20-%20Schoolboy%20Facination.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FreGrad/Al%20James%20-%20Schoolboy%20Facination.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FastDiff/Al%20James%20-%20Schoolboy%20Facination.wav"></audio>
      </td>
    </tr>
  </tbody>
  <tbody>
    <tr><td colspan="8">Music 2 (out of distribution)</td></tr>
  </tbody>
  <tbody>
    <tr>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/GT/Alexander%20Ross%20-%20Goodbye%20Bolero.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM1/Alexander%20Ross%20-%20Goodbye%20Bolero.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM6/Alexander%20Ross%20-%20Goodbye%20Bolero.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/HifiGAN/Alexander%20Ross%20-%20Goodbye%20Bolero.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/DiffWave/Alexander%20Ross%20-%20Goodbye%20Bolero.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/PriorGrad/Alexander%20Ross%20-%20Goodbye%20Bolero.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FreGrad/Alexander%20Ross%20-%20Goodbye%20Bolero.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FastDiff/Alexander%20Ross%20-%20Goodbye%20Bolero.wav"></audio>
      </td>
    </tr>
  </tbody>
  <tbody>
    <tr><td colspan="8">Bass (out of distribution)</td></tr>
  </tbody>
  <tbody>
    <tr>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/GT/A%20Classic%20Education%20-%20NightOwl.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM1/A%20Classic%20Education%20-%20NightOwl.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/WaveFM6/A%20Classic%20Education%20-%20NightOwl.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/HifiGAN/A%20Classic%20Education%20-%20NightOwl.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/DiffWave/A%20Classic%20Education%20-%20NightOwl.wav"></audio>
      </td>
      <td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/PriorGrad/A%20Classic%20Education%20-%20NightOwl.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FreGrad/A%20Classic%20Education%20-%20NightOwl.wav"></audio>
      </td><td align="center">
        <audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio/FastDiff/A%20Classic%20Education%20-%20NightOwl.wav"></audio>
      </td>
    </tr>
  </tbody>
</table>
