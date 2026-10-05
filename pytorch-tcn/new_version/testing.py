from newnet import ECG5000FullQuantTCN
import torch

def qt_value(x):
    return x.value if hasattr(x, "value") else x

model = ECG5000FullQuantTCN(
    num_classes=5,
    seq_len=140,
    num_inputs=1,
    num_channels=(16, 16, 32),
    kernel_size=3,
    dropout=0.1,
    causal=True,
)

model.eval()
x = torch.zeros(1, 1, 140, 1)

with torch.no_grad():
    y = qt_value(model(x))

print(y.shape)
print(y)