import torch
import torch.nn as nn

#from quant_tcn.test_model import ToyQuantTCN

from quant_tcn.quant_tcn_test import QTCN
from quant_tcn.quant_buffer import BufferIO



import brevitas.nn as qnn
from brevitas.quant import Int8WeightPerTensorFloat, Int32Bias

class ECG5000QModel(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        self.backbone = QTCN(
            num_inputs=1,
            num_channels=[16, 16, 32],
            kernel_size=3,
            dropout=0.1,
            causal=False,
            use_norm=None,
            activation='relu',
            kernel_initializer='xavier_uniform',
            use_skip_connections=False,
            input_shape='NCL',
            output_projection=None,
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = qnn.QuantLinear(
            in_features=32,
            out_features=num_classes,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            return_quant_tensor=False,
        )

    def forward(self, x):
        # x: [N, 1, L]
        x = self.backbone(x)            # [N, 32, L]
        x = self.pool(x).squeeze(-1)    # [N, 32]
        x = self.classifier(x)          # [N, 5]
        return x
