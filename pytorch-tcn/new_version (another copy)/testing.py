import torch
import numpy as np
from newnet import ECG5000FullQuantTCN


NUM_CLASSES = 5
SEQ_LEN = 140
NUM_INPUTS = 1
NUM_CHANNELS = (16, 16, 32)
KERNEL_SIZE = 3
DROPOUT = 0.1
CAUSAL = True

BATCH_SIZE = 64
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 20

USE_CLASS_WEIGHT = True
SAVE_NAME = "best_qtcn_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ECG5000FullQuantTCN(
        num_classes=NUM_CLASSES,
        seq_len=SEQ_LEN,
        num_inputs=NUM_INPUTS,
        num_channels=NUM_CHANNELS,
        kernel_size=KERNEL_SIZE,
        dropout=DROPOUT,
        causal=CAUSAL,
    ).to(DEVICE)

model.eval()

x = torch.tensor(
    [[[[
        3.6908443],
        [0.71141434],
        [-2.1140914],
        [-4.141007],
        [-4.5744715],
    ]]],
    dtype=torch.float32,
)

with torch.no_grad():
    q = model.input_quant(x)

print("q type:", type(q))
print("q value:")
print(q.value.reshape(-1))

print("q scale:")
print(q.scale)

print("q zero_point:")
print(q.zero_point)

print("q bit_width:")
print(q.bit_width)

q_int = torch.round(q.value / q.scale + q.zero_point)
print("q int:")
print(q_int.reshape(-1))