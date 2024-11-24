---
layout: default
---

Flow matching offers a robust and stable approach to training diffusion models. However, directly applying flow matching to neural vocoders can result in subpar audio quality. In this work, we present WaveFM, a reparameterized flow matching model for mel-spectrogram conditioned speech synthesis, designed to enhance both sample quality and generation speed for diffusion vocoders. Since mel-spectrograms represent the energy distribution of waveforms, WaveFM adopts a mel-conditioned prior distribution instead of a standard Gaussian prior to minimize unnecessary transportation costs during synthesis. Moreover, while most diffusion vocoders rely on a single loss function, we argue that incorporating auxiliary losses, including a refined multi-resolution STFT loss, can further improve audio quality. To speed up inference without degrading sample quality significantly, we introduce a tailored consistency distillation method for WaveFM. Experimental results demonstrate that our model achieves superior performance in both quality and efficiency compared to previous diffusion vocoders, while enabling waveform generation in a single inference step.

# Model

![Model](./model.png)

The total amount of parameters is `19.5M`. The 128-dim time embedding is expanded to 512-dim after two linear-SiLU layers, and is then reshaped to the desired shape of each resolution. `Conv1d` and `ConvTranspose1d` are set with parameters `(output channel, kernel width, dilation, padding)`. In ResBlock `Conv1d` takes same padding. Each ResLayer is defined with a kernel list and a dilation list, and their cross-product of define the shape of the ResBlock matrix and the convolutional layers of each ResBlock. On the left column are downsampling ResLayers, each containing a `4 x 1` ResBlock matrix, while on the right columns are upsampling ResLayers, each containing a `3 x 3` ResBlock matrix, following the structure from HifiGAN. In each ResBlock the number of channels is unchanged.

For detailed parameter settings please refer to `WaveFM/src/params.py`.

# Audio Samples

<table><thead><tr><td align="center"><b>Ground</b><br><b>Truth</b></td>
<td align="center"><b>WaveFM</b><br><b>(6 steps)</b></td>
<td align="center"><b>WaveFM</b><br><b>(1 step)</b></td>
<td align="center"><b>BigVGAN</b><br><b>(6 steps)</b></td>
<td align="center"><b>HifiGAN</b><br><b>(6 steps)</b></td>
<td align="center"><b>DiffWave</b><br><b>(6 steps)</b></td>
<td align="center"><b>PriorGrad</b><br><b>(6 steps)</b></td>
<td align="center"><b>FreGrad</b><br><b>(6 steps)</b></td>
<td align="center"><b>FastDiff</b><br><b>(6 steps)</b></td></tr></thead><tbody>
<tbody><tr><td colspan="9">MUSDB18-HQ Mixture 1</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\Actions_-_South_Of_The_Water.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\Actions_-_South_Of_The_Water.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\Actions_-_South_Of_The_Water.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\Actions_-_South_Of_The_Water.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\Actions_-_South_Of_The_Water.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\Actions_-_South_Of_The_Water.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\Actions_-_South_Of_The_Water.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\Actions_-_South_Of_The_Water.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\Actions_-_South_Of_The_Water.wav"></audio></td>
</tr></tbody><tbody><tr><td colspan="9">MUSDB18-HQ Mixture 2</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\Enda_Reilly_-_Cur_An_Long_Ag_Seol.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\Enda_Reilly_-_Cur_An_Long_Ag_Seol.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\Enda_Reilly_-_Cur_An_Long_Ag_Seol.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\Enda_Reilly_-_Cur_An_Long_Ag_Seol.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\Enda_Reilly_-_Cur_An_Long_Ag_Seol.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\Enda_Reilly_-_Cur_An_Long_Ag_Seol.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\Enda_Reilly_-_Cur_An_Long_Ag_Seol.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\Enda_Reilly_-_Cur_An_Long_Ag_Seol.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\Enda_Reilly_-_Cur_An_Long_Ag_Seol.wav"></audio></td>
</tr></tbody><tbody><tr><td colspan="9">MUSDB18-HQ Mixture 3</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\James_May_-_If_You_Say.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\James_May_-_If_You_Say.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\James_May_-_If_You_Say.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\James_May_-_If_You_Say.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\James_May_-_If_You_Say.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\James_May_-_If_You_Say.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\James_May_-_If_You_Say.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\James_May_-_If_You_Say.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\James_May_-_If_You_Say.wav"></audio></td>
</tr></tbody><tbody><tr><td colspan="9">MUSDB18-HQ Mixture 4</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\Mu_-_Too_Bright.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\Mu_-_Too_Bright.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\Mu_-_Too_Bright.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\Mu_-_Too_Bright.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\Mu_-_Too_Bright.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\Mu_-_Too_Bright.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\Mu_-_Too_Bright.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\Mu_-_Too_Bright.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\Mu_-_Too_Bright.wav"></audio></td>
</tr></tbody><tbody><tr><td colspan="9">MUSDB18-HQ Mixture 5</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\Skelpolu_-_Resurrection.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\Skelpolu_-_Resurrection.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\Skelpolu_-_Resurrection.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\Skelpolu_-_Resurrection.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\Skelpolu_-_Resurrection.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\Skelpolu_-_Resurrection.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\Skelpolu_-_Resurrection.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\Skelpolu_-_Resurrection.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\Skelpolu_-_Resurrection.wav"></audio></td>
</tr></tbody><tbody><tr><td colspan="9">MUSDB18-HQ Mixture 6</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\Young_Griffo_-_Facade.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\Young_Griffo_-_Facade.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\Young_Griffo_-_Facade.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\Young_Griffo_-_Facade.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\Young_Griffo_-_Facade.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\Young_Griffo_-_Facade.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\Young_Griffo_-_Facade.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\Young_Griffo_-_Facade.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\Young_Griffo_-_Facade.wav"></audio></td>
</tr></tbody><tbody><tr><td colspan="9">MUSDB18-HQ Bass 1</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\Actions_-_Devil's_Words.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\Actions_-_Devil's_Words.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\Actions_-_Devil's_Words.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\Actions_-_Devil's_Words.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\Actions_-_Devil's_Words.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\Actions_-_Devil's_Words.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\Actions_-_Devil's_Words.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\Actions_-_Devil's_Words.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\Actions_-_Devil's_Words.wav"></audio></td>
</tr></tbody><tbody><tr><td colspan="9">MUSDB18-HQ Drum 1</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\Leaf_-_Summerghost.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\Leaf_-_Summerghost.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\Leaf_-_Summerghost.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\Leaf_-_Summerghost.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\Leaf_-_Summerghost.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\Leaf_-_Summerghost.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\Leaf_-_Summerghost.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\Leaf_-_Summerghost.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\Leaf_-_Summerghost.wav"></audio></td>
</tr></tbody><tbody><tr><td colspan="9">MUSDB18-HQ Vocal 1</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\Flags_-_54.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\Flags_-_54.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\Flags_-_54.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\Flags_-_54.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\Flags_-_54.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\Flags_-_54.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\Flags_-_54.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\Flags_-_54.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\Flags_-_54.wav"></audio></td>
</tr></tbody><tbody><tr><td colspan="9">MUSDB18-HQ Vocal 2</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\The_Wrong'Uns_-_Rothko.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\The_Wrong'Uns_-_Rothko.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\The_Wrong'Uns_-_Rothko.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\The_Wrong'Uns_-_Rothko.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\The_Wrong'Uns_-_Rothko.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\The_Wrong'Uns_-_Rothko.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\The_Wrong'Uns_-_Rothko.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\The_Wrong'Uns_-_Rothko.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\The_Wrong'Uns_-_Rothko.wav"></audio></td>
</tr></tbody><tbody><tr><td colspan="9">MUSDB18-HQ Vocal 3</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\Bill_Chudziak_-_Children_Of_No-one.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\Bill_Chudziak_-_Children_Of_No-one.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\Bill_Chudziak_-_Children_Of_No-one.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\Bill_Chudziak_-_Children_Of_No-one.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\Bill_Chudziak_-_Children_Of_No-one.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\Bill_Chudziak_-_Children_Of_No-one.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\Bill_Chudziak_-_Children_Of_No-one.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\Bill_Chudziak_-_Children_Of_No-one.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\Bill_Chudziak_-_Children_Of_No-one.wav"></audio></td>
</tr></tbody><tbody><tr><td colspan="9">MUSDB18-HQ Others 1</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\Fergessen_-_Nos_Palpitants.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\Fergessen_-_Nos_Palpitants.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\Fergessen_-_Nos_Palpitants.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\Fergessen_-_Nos_Palpitants.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\Fergessen_-_Nos_Palpitants.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\Fergessen_-_Nos_Palpitants.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\Fergessen_-_Nos_Palpitants.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\Fergessen_-_Nos_Palpitants.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\Fergessen_-_Nos_Palpitants.wav"></audio></td>
</tr></tbody><tbody><tr><td colspan="9">LibriTTS Test 1</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\84_121123_000015_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\84_121123_000015_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\84_121123_000015_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\84_121123_000015_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\84_121123_000015_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\84_121123_000015_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\84_121123_000015_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\84_121123_000015_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\84_121123_000015_000000.wav"></audio></td>
</tr></tbody><tbody><tr><td colspan="9">LibriTTS Test 2</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\1272_135031_000054_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\1272_135031_000054_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\1272_135031_000054_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\1272_135031_000054_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\1272_135031_000054_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\1272_135031_000054_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\1272_135031_000054_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\1272_135031_000054_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\1272_135031_000054_000000.wav"></audio></td>
</tr></tbody><tbody><tr><td colspan="9">LibriTTS Test 3</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\1673_143397_000031_000011.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\1673_143397_000031_000011.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\1673_143397_000031_000011.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\1673_143397_000031_000011.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\1673_143397_000031_000011.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\1673_143397_000031_000011.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\1673_143397_000031_000011.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\1673_143397_000031_000011.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\1673_143397_000031_000011.wav"></audio></td>
</tr></tbody><tbody><tr><td colspan="9">LibriTTS Test 4</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\174_168635_000024_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\174_168635_000024_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\174_168635_000024_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\174_168635_000024_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\174_168635_000024_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\174_168635_000024_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\174_168635_000024_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\174_168635_000024_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\174_168635_000024_000001.wav"></audio></td>
</tr></tbody><tbody><tr><td colspan="9">LibriTTS Test 5</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\2277_149896_000023_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\2277_149896_000023_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\2277_149896_000023_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\2277_149896_000023_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\2277_149896_000023_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\2277_149896_000023_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\2277_149896_000023_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\2277_149896_000023_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\2277_149896_000023_000001.wav"></audio></td>
</tr></tbody><tbody><tr><td colspan="9">LibriTTS Test 6</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\3752_4944_000062_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\3752_4944_000062_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\3752_4944_000062_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\3752_4944_000062_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\3752_4944_000062_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\3752_4944_000062_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\3752_4944_000062_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\3752_4944_000062_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\3752_4944_000062_000000.wav"></audio></td>
</tr></tbody><tbody><tr><td colspan="9">LibriTTS Test 7</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\3576_138058_000048_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\3576_138058_000048_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\3576_138058_000048_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\3576_138058_000048_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\3576_138058_000048_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\3576_138058_000048_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\3576_138058_000048_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\3576_138058_000048_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\3576_138058_000048_000000.wav"></audio></td>
</tr></tbody><tbody><tr><td colspan="9">LibriTTS Test 8</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\5338_284437_000037_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\5338_284437_000037_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\5338_284437_000037_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\5338_284437_000037_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\5338_284437_000037_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\5338_284437_000037_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\5338_284437_000037_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\5338_284437_000037_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\5338_284437_000037_000001.wav"></audio></td>
</tr></tbody><tbody><tr><td colspan="9">LibriTTS Test 9</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\5536_43358_000011_000002.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\5536_43358_000011_000002.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\5536_43358_000011_000002.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\5536_43358_000011_000002.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\5536_43358_000011_000002.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\5536_43358_000011_000002.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\5536_43358_000011_000002.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\5536_43358_000011_000002.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\5536_43358_000011_000002.wav"></audio></td>
</tr></tbody><tbody><tr><td colspan="9">LibriTTS Test 10</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\6241_61946_000049_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\6241_61946_000049_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\6241_61946_000049_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\6241_61946_000049_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\6241_61946_000049_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\6241_61946_000049_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\6241_61946_000049_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\6241_61946_000049_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\6241_61946_000049_000000.wav"></audio></td>
</tr></tbody><tbody><tr><td colspan="9">LibriTTS Test 11</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\7850_73752_000010_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\7850_73752_000010_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\7850_73752_000010_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\7850_73752_000010_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\7850_73752_000010_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\7850_73752_000010_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\7850_73752_000010_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\7850_73752_000010_000000.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\7850_73752_000010_000000.wav"></audio></td>
</tr></tbody><tbody><tr><td colspan="9">LibriTTS Test 12</td></tr></tbody><tbody><tr>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\Ground_Truth\8297_275154_000030_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(6_steps)\8297_275154_000030_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\WaveFM_(1_step)\8297_275154_000030_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\BigVGAN_(6_steps)\8297_275154_000030_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\HifiGAN_(6_steps)\8297_275154_000030_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\DiffWave_(6_steps)\8297_275154_000030_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\PriorGrad_(6_steps)\8297_275154_000030_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FreGrad_(6_steps)\8297_275154_000030_000001.wav"></audio></td>
<td align="center"><audio id="player" controls="" style="width:100px;" preload="auto"><source src="audio\FastDiff_(6_steps)\8297_275154_000030_000001.wav"></audio></td>
</tr></tbody>