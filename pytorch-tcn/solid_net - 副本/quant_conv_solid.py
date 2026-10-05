import torch
import torch.nn as nn
import brevitas.nn as qnn

from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias


class TemporalConv2dExport(nn.Module):
    """
    4D causal temporal conv for FINN-friendly export.

    输入:
        x: [N, C_in, 1, L]

    输出:
        y: [N, C_out, 1, L_out]

    说明:
        - 整个 backbone 内部统一保持 4D
        - 左侧 causal padding 也在 4D 上做
        - 不在层间做 squeeze / unsqueeze
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        input_quant=Int8ActPerTensorFloat,
        weight_quant=Int8WeightPerTensorFloat,
        bias_quant=Int32Bias,
        return_quant_tensor: bool = False,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.dilation = int(dilation)
        self.groups = int(groups)
        self.pad_len = (self.kernel_size - 1) * self.dilation

        # Pad2d 的 pad 顺序: (left, right, top, bottom)
        self.left_pad = nn.ConstantPad2d((self.pad_len, 0, 0, 0), 0.0)

        self.conv = qnn.QuantConv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(1, self.kernel_size),
            stride=(1, self.stride),
            padding=0,
            dilation=(1, self.dilation),
            groups=self.groups,
            bias=bias,
            input_quant=None,
            weight_quant=weight_quant,
            bias_quant=None,
            return_quant_tensor=return_quant_tensor,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, C, 1, L]
        orig_len = x.shape[-1]

        x = self.left_pad(x)
        x = self.conv(x)

        # 强制裁回原始时间长度，去掉左侧因 causal padding 带来的多余位置
        x = x[..., -orig_len:]
        return x


class PointwiseConv2dExport(nn.Module):
    """
    4D pointwise conv for residual / projection path.

    输入:
        x: [N, C_in, 1, L]

    输出:
        y: [N, C_out, 1, L]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bias: bool = True,
        input_quant=Int8ActPerTensorFloat,
        weight_quant=Int8WeightPerTensorFloat,
        bias_quant=Int32Bias,
        return_quant_tensor: bool = False,
    ):
        super().__init__()

        self.conv = qnn.QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            dilation=(1, 1),
            groups=1,
            bias=bias,
            input_quant=None,
            weight_quant=weight_quant,
            bias_quant=None,
            return_quant_tensor=return_quant_tensor,
        )

    @property
    def weight(self):
        return self.conv.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)