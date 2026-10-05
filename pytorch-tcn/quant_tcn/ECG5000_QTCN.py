import torch
import torch.nn as nn
import brevitas.nn as qnn

from brevitas.quant import Uint8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias


class StaticQResBlock2D(nn.Module):
    """
    输入/输出张量格式: [N, C, 1, 140]
    用 QuantConv2d 的 (1, k) 卷积去模拟 1D TCN
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 必须是奇数"
        pad = dilation * (kernel_size - 1) // 2

        conv_kwargs = dict(
            weight_quant=Int8WeightPerTensorFloat,
            bias=False,
            bias_quant=Int32Bias,
            return_quant_tensor=False,
        )

        self.conv1 = qnn.QuantConv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, kernel_size),
            stride=1,
            padding=(0, pad),
            dilation=(1, dilation),
            **conv_kwargs,
        )
        self.act1 = qnn.QuantReLU(
            act_quant=Uint8ActPerTensorFloat,
            return_quant_tensor=False,
        )

        self.conv2 = qnn.QuantConv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=(1, kernel_size),
            stride=1,
            padding=(0, pad),
            dilation=(1, dilation),
            **conv_kwargs,
        )

        if in_ch != out_ch:
            self.skip = qnn.QuantConv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=1,
                stride=1,
                padding=0,
                **conv_kwargs,
            )
        else:
            self.skip = nn.Identity()

        self.out_act = qnn.QuantReLU(
            act_quant=Uint8ActPerTensorFloat,
            return_quant_tensor=False,
        )

    def forward(self, x):
        residual = self.skip(x)

        x = self.conv1(x)
        x = self.act1(x)

        x = self.conv2(x)
        x = x + residual
        x = self.out_act(x)

        return x


class ECG5000QTCNClassifier(nn.Module):
    """
    静态部署版：
    输入固定:  [N, 1, 140]
    输出固定:  [N, 5]
    """

    def __init__(self, num_classes=5):
        super().__init__()

        self.in_quant = qnn.QuantIdentity(
            act_quant=Uint8ActPerTensorFloat,
            return_quant_tensor=False,
        )

        # 3 个 TCN block，对应你之前常用的 (16, 16, 32)
        self.block1 = StaticQResBlock2D(1, 16, kernel_size=3, dilation=1)
        self.block2 = StaticQResBlock2D(16, 16, kernel_size=3, dilation=2)
        self.block3 = StaticQResBlock2D(16, 32, kernel_size=3, dilation=4)

        # 不用 adaptive pool，直接静态 flatten
        # 经过上面 3 个 block 后，空间尺寸仍然固定为 [1, 140]
        self.flatten = nn.Flatten(start_dim=1)

        self.fc1 = qnn.QuantLinear(
            in_features=32 * 1 * 140,
            out_features=64,
            weight_quant=Int8WeightPerTensorFloat,
            bias=False,
            bias_quant=Int32Bias,
            return_quant_tensor=False,
        )
        self.fc1_act = qnn.QuantReLU(
            act_quant=Uint8ActPerTensorFloat,
            return_quant_tensor=False,
        )

        self.fc2 = qnn.QuantLinear(
            in_features=64,
            out_features=num_classes,
            weight_quant=Int8WeightPerTensorFloat,
            bias=False,
            bias_quant=Int32Bias,
            return_quant_tensor=False,
        )

    def forward(self, x):
        # x: [N, 1, 140]
        x = self.in_quant(x)

        # 改成 2D 卷积输入: [N, 1, 1, 140]
        x = x.unsqueeze(2)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.flatten(x)   # [N, 32*140]
        x = self.fc1(x)
        x = self.fc1_act(x)
        x = self.fc2(x)       # [N, 5]

        return x