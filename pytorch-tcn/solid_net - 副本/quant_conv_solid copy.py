import torch
import torch.nn as nn
import brevitas.nn as qnn

from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias

from solid_net.quant_pad_solid import TemporalPad1dExport


class PointwiseConv1dExport(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=True,
        input_quant=Int8ActPerTensorFloat,
        weight_quant=Int8WeightPerTensorFloat,
        bias_quant=Int32Bias,
        return_quant_tensor=False,
    ):
        super().__init__()
        self.conv = qnn.QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            bias=bias,
            input_quant=input_quant,
            weight_quant=weight_quant,
            bias_quant=bias_quant if bias else None,
            return_quant_tensor=return_quant_tensor,
        )

    def forward(self, x):
        x = x.unsqueeze(2)   # [N, C, 1, L]
        x = self.conv(x)     # [N, C_out, 1, L]
        x = x[:, :, 0, :]    # [N, C_out, L]
        return x

class TemporalConv1dExport(nn.Module):
    """
    Export-only causal TemporalConv1d.

    输入:
        x: [B, C_in, T]

    输出:
        y: [B, C_out, T_out]
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
        device=None,
        dtype=None,
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

        self.padder = TemporalPad1dExport(
            padding=self.pad_len,
            in_channels=self.in_channels,
            device=device,
            dtype=dtype,
        )

        self.conv = qnn.QuantConv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=(1, self.kernel_size),
            stride=(1, self.stride),
            padding=0,
            dilation=(1, self.dilation),
            groups=self.groups,
            bias=bias,
            input_quant=input_quant,
            weight_quant=weight_quant,
            bias_quant=bias_quant if bias else None,
            return_quant_tensor=return_quant_tensor,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.padder(x)       # [B, C, T + pad_len]
        x = x.unsqueeze(2)       # [B, C, 1, T + pad_len]
        x = self.conv(x)         # [B, C_out, 1, T_out]
        x = x[:, :, 0, :]        # [B, C_out, T_out]
        return x