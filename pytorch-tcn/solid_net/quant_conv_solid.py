import torch
import torch.nn as nn
import brevitas.nn as qnn

from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias

from .quant_pad_solid import TemporalPad2dTrain, TemporalPad2dExport


class PointwiseConv1dCore(nn.Module):
    """
    4D pointwise temporal conv.

    输入:
        x: [B, C_in, 1, T]

    输出:
        y: [B, C_out, 1, T]
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        bias=False,
        input_quant=Int8ActPerTensorFloat,   # 保留接口兼容，不实际使用
        weight_quant=Int8WeightPerTensorFloat,
        bias_quant=Int32Bias,
        return_quant_tensor=True,
    ):
        super().__init__()

        self.conv = qnn.QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            bias=bias,
            input_quant=None,
            weight_quant=weight_quant,
            bias_quant=bias_quant if bias else None,
            return_quant_tensor=return_quant_tensor,
        )

    def forward(self, x: torch.Tensor):
        if x.dim() != 4:
            raise ValueError(
                f"PointwiseConv1dCore expects 4D input [B, C, 1, T], got shape {tuple(x.shape)}"
            )
        if x.shape[2] != 1:
            raise ValueError(
                f"PointwiseConv1dCore expects height dimension == 1, got shape {tuple(x.shape)}"
            )

        return self.conv(x)


class TemporalConv1dCore(nn.Module):
    """
    4D temporal conv core.

    输入:
        x: [B, C_in, 1, T]

    输出:
        y: [B, C_out, 1, T_out]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        device=None,
        dtype=None,
        input_quant=Int8ActPerTensorFloat,   # 保留接口兼容，不实际使用
        weight_quant=Int8WeightPerTensorFloat,
        bias_quant=Int32Bias,
        return_quant_tensor: bool = True,
        padder_cls=TemporalPad2dTrain,
    ):
        super().__init__()

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.dilation = int(dilation)
        self.groups = int(groups)
        self.pad_len = (self.kernel_size - 1) * self.dilation

        self.padder = padder_cls(
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
            input_quant=None,
            weight_quant=weight_quant,
            bias_quant=bias_quant if bias else None,
            return_quant_tensor=return_quant_tensor,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"TemporalConv1dCore expects 4D input [B, C, 1, T], got shape {tuple(x.shape)}"
            )
        if x.shape[2] != 1:
            raise ValueError(
                f"TemporalConv1dCore expects height dimension == 1, got shape {tuple(x.shape)}"
            )

        x = self.padder(x)   # [B, C, 1, T + pad_len]
        x = self.conv(x)     # [B, C_out, 1, T_out]
        return x


# -----------------------------------------------------------------------------
# 兼容你现在已有的 import 名字
# -----------------------------------------------------------------------------

class PointwiseConv1dExport(PointwiseConv1dCore):
    pass


class TemporalConv1dExport(TemporalConv1dCore):
    def __init__(self, *args, **kwargs):
        # 如果外面没显式传 padder_cls，就默认走训练友好的动态 batch 版本
        kwargs.setdefault("padder_cls", TemporalPad2dTrain)
        super().__init__(*args, **kwargs)


def make_export_temporal_conv(*args, **kwargs):
    """
    导出专用快捷构造器：
    固定使用静态 batch=1 的 padder
    """
    kwargs["padder_cls"] = TemporalPad2dExport
    return TemporalConv1dCore(*args, **kwargs)