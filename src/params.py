params = dict(
    # params for mel-spectrograms
    fftSize=1024,
    windowSize=1024,
    hopSize=256,  # don't change STFT's hopsize
    melBands=100,
    sampleRate=24000,
    fmin=0,
    fmax=12000,
    melTrainWindow=64,  # The shape of training audios = batch * 1 * (melTrainWindow*hopSize)

    # params for training
    trainWithHybridPrecision=False,
    trainInitialLR=1e-4,
    trainFinalLR=5e-6,
    trainBetas=(0.9, 0.99),
    trainWeightDecay=5e-4,
    trainSteps=1000000,
    trainBatch=16,
    trainCheckPointSavingStep=10000,

    trainAudiosPath=r"Your audios",
    trainMelsPath=r"Your mel-spectrograms",
    trainCheckPointPath="./checkpoints/WaveFM_0",
    trainDevice="cuda:0",

    # params for distillation (training hyperparameters and data are kept)
    distillWithHybridPrecision=False,
    distillInitialLR=2e-5,
    distillFinalLR=5e-6,
    distillBetas=(0.8, 0.95),
    distillWeightDecay=1e-2,
    distillDeltaT=0.01,
    distillSteps=25000,
    distillBatch=16,
    distillCheckPointSavingStep=5000,
    
    distillModelPath= './checkpoints/WaveFM_1000000',
    distillCheckPointPath='./distilled_checkpoints/Distilled_WaveFM_0',
    distillAudiosPath=r"Your audios",
    distillMelsPath=r"Your mel-spectrograms",
    distillDevice="cuda:0",

    # params for inference
    inferenceSteps=6,
    
    inferenceMelsPath=r"Your inference mel-spectrograms",
    inferenceSavingPath="./generation",
    inferenceCheckPointPath=
    './checkpoints/WaveFM_1000000',
    #'./distilled_checkpoints/Distilled_WaveFM_25000',
    inferenceDevice="cuda:0",

    # params for model
    timeEmbeddingSize=512,
    velocityChannels=[512, 256, 128, 64, 32],
    velocityUpSampleRates=[8, 8, 2, 2],
    velocityKernelSizesUp=[[3, 7, 11], [3, 7, 11], [3, 7, 11], [3, 7, 11]],
    velocityDilationsUp=[[1, 3, 5], [1, 3, 5], [1, 3, 5], [1, 3, 5]],
    velocityKernelSizesDown=[[3], [3], [3], [3]],
    velocityDilationsDown=[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
    
)
