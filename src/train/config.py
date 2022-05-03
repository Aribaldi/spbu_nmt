import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMB_SIZE = 512
NHEAD = 4
FFN_HID_DIM = 512
BATCH_SIZE = 8 #128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
NUM_WORKERS = 2
TRAIN_LEN = 225920 #29000
VAL_LEN = 2048 #1014