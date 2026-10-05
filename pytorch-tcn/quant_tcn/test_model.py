import torch
import torch.nn as nn
import brevitas.nn as qnn

from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias

from .quant_conv import TemporalConv1d
# 如果你的文件名不是 quant_conv，就改成你自己的模块名


class ToyQuantTCN(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 5,
        hidden_channels: int = 16,
        kernel_size: int = 3,
        dilation1: int = 1,
        dilation2: int = 2,
        causal: bool = True,
    ):
        super().__init__()

        # 第一层量化时序卷积
        self.conv1 = TemporalConv1d(
            in_channels=input_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation1,
            causal=causal,
            input_quant=Int8ActPerTensorFloat,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            return_quant_tensor=False,
        )

        # 量化激活
        self.act1 = qnn.QuantReLU(
            bit_width=8,
            return_quant_tensor=False,
        )

        # 第二层量化时序卷积
        self.conv2 = TemporalConv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            dilation=dilation2,
            causal=causal,
            input_quant=Int8ActPerTensorFloat,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            return_quant_tensor=False,
        )

        # 第二个量化激活
        self.act2 = qnn.QuantReLU(
            bit_width=8,
            return_quant_tensor=False,
        )

        # 最后一层分类器
        # 这里输入是最后一个时间步上的 hidden_channels 维特征
        self.fc = qnn.QuantLinear(
            in_features=hidden_channels,
            out_features=num_classes,
            bias=True,
            input_quant=Int8ActPerTensorFloat,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            return_quant_tensor=False,
        )

    def forward(self, x: torch.Tensor):
        """
        x shape: [B, C, T]
        e.g. ECG5000 若单通道，可设为 [B, 1, T]
        """
        x = self.conv1(x)      # [B, H, T]
        x = self.act1(x)

        x = self.conv2(x)      # [B, H, T]
        x = self.act2(x)

        # 取最后一个时间步做分类
        # shape: [B, H]
        x = x[:, :, -1]

        x = self.fc(x)         # [B, num_classes]
        return x
