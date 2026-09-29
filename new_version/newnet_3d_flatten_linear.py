from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import brevitas.nn as qnn

from brevitas.quant.fixed_point import (
    Int8WeightPerTensorFixedPoint,
    Int8ActPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
    Int8BiasPerTensorFixedPointInternalScaling,
)


# ============================================================
# Quantizer definitions
# ============================================================

class Int4ActPerTensorFixedPoint(Int8ActPerTensorFixedPoint):
    bit_width = 4


class Uint4ActPerTensorFixedPoint(Uint8ActPerTensorFixedPoint):
    bit_width = 4


class Int2WeightPerTensorFixedPoint(Int8WeightPerTensorFixedPoint):
    bit_width = 2


class Int4WeightPerTensorFixedPoint(Int8WeightPerTensorFixedPoint):
    bit_width = 4


class Int2ActPerTensorFixedPoint(Int8ActPerTensorFixedPoint):
    bit_width = 2


class Uint2ActPerTensorFixedPoint(Uint8ActPerTensorFixedPoint):
    bit_width = 2


class Int1ActPerTensorFixedPoint(Int8ActPerTensorFixedPoint):
    bit_width = 1


class Uint1ActPerTensorFixedPoint(Uint8ActPerTensorFixedPoint):
    bit_width = 1


def qt_value(x):
    """Return the underlying tensor if x is a Brevitas QuantTensor."""
    return x.value if hasattr(x, "value") else x


def qt_set(x, value):
    """Replace the tensor value inside a Brevitas QuantTensor."""
    if hasattr(x, "set"):
        return x.set(value=value)
    raise TypeError("Expected QuantTensor input.")


# ============================================================
# 3D feature causal TCN blocks
# Feature layout:
#     x: [N, C, L]
# ============================================================

class CausalPad1dExport(nn.Module):
    """
    Causal/asymmetric padding for 3D temporal features.

    Input:
        x: [N, C, L]

    F.pad for 3D sequence tensors uses:
        (left, right)
    """
    def __init__(self, pad_left: int, pad_right: int = 0):
        super().__init__()
        self.pad_left = pad_left
        self.pad_right = pad_right

    def forward(self, x):
        y = F.pad(qt_value(x), (self.pad_left, self.pad_right), "constant", 0.0)
        return qt_set(x, y)


class QuantTemporalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
        causal: bool = True,
    ):
        super().__init__()

        if causal:
            self.pad_left = (kernel_size - 1) * dilation
            self.pad_right = 0
        else:
            total_pad = (kernel_size - 1) * dilation
            self.pad_left = total_pad // 2
            self.pad_right = total_pad - self.pad_left

        self.pad = CausalPad1dExport(self.pad_left, self.pad_right)

        self.conv = qnn.QuantConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            dilation=dilation,
            bias=bias,
            weight_quant=Int4WeightPerTensorFixedPoint,
            bias_quant=Int8BiasPerTensorFixedPointInternalScaling,
            return_quant_tensor=True,
        )

    def forward(self, x):
        # x: [N, C, L]
        x = self.pad(x)
        x = self.conv(x)
        return x


class QuantTemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()

        self.drop1_p = dropout
        self.drop2_p = dropout

        self.conv1 = QuantTemporalConv1d(
            in_channels=n_inputs,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=True,
            causal=causal,
        )

        self.act1 = qnn.QuantReLU(
            act_quant=Uint4ActPerTensorFixedPoint,
            return_quant_tensor=True,
        )

        self.conv2 = QuantTemporalConv1d(
            in_channels=n_outputs,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=True,
            causal=causal,
        )

        if n_inputs != n_outputs:
            self.downsample = qnn.QuantConv1d(
                in_channels=n_inputs,
                out_channels=n_outputs,
                kernel_size=1,
                stride=1,
                bias=True,
                weight_quant=Int4WeightPerTensorFixedPoint,
                bias_quant=Int8BiasPerTensorFixedPointInternalScaling,
                return_quant_tensor=True,
            )
        else:
            self.downsample = None

        self.pre_add_quant = qnn.QuantIdentity(
            act_quant=Int4ActPerTensorFixedPoint,
            return_quant_tensor=True,
        )

        self.post_add_act = qnn.QuantReLU(
            act_quant=Uint4ActPerTensorFixedPoint,
            return_quant_tensor=True,
        )

    def forward(self, x):
        res = x if self.downsample is None else self.downsample(x)

        out = self.conv1(x)
        out = self.act1(out)

        if self.drop1_p > 0:
            out = qt_set(
                out,
                F.dropout(qt_value(out), p=self.drop1_p, training=self.training),
            )

        out = self.conv2(out)

        if self.drop2_p > 0:
            out = qt_set(
                out,
                F.dropout(qt_value(out), p=self.drop2_p, training=self.training),
            )

        # Make the residual and main branch use the same activation quantizer
        # before the QuantTensor addition.
        out = self.pre_add_quant(qt_value(out))
        res = self.pre_add_quant(qt_value(res))

        out = out + res
        out = self.post_add_act(out)
        return out


# ============================================================
# ECG5000 TCN with 3D features + Flatten + QuantLinear head
# Input:
#     [N, C, L], e.g. [N, 1, 140]
# Output:
#     [N, num_classes], e.g. [N, 5]
# ============================================================

class ECG5000FullQuantTCN(nn.Module):
    def __init__(
        self,
        num_classes: int = 5,
        seq_len: int = 140,
        num_inputs: int = 1,
        num_channels=(16, 16, 16),
        kernel_size: int = 5,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()

        self.seq_len = seq_len
        self.num_channels = tuple(num_channels)

        self.input_quant = qnn.QuantIdentity(
            act_quant=Int4ActPerTensorFixedPoint,
            return_quant_tensor=True,
        )

        layers = []
        in_ch = num_inputs
        for i, out_ch in enumerate(num_channels):
            print("in_ch =", in_ch)
            print("out_ch =", out_ch)

            layers.append(
                QuantTemporalBlock(
                    n_inputs=in_ch,
                    n_outputs=out_ch,
                    kernel_size=kernel_size,
                    # Use dilation=2 ** i if you want the original TCN dilation pattern.
                    dilation=1,
                    dropout=dropout,
                    causal=causal,
                )
            )
            in_ch = out_ch

        self.backbone = nn.Sequential(*layers)

        self.flatten = nn.Flatten(start_dim=1)

        self.classifier = qnn.QuantLinear(
            in_features=num_channels[-1] * seq_len,
            out_features=num_classes,
            bias=True,
            input_quant=Int4ActPerTensorFixedPoint,
            weight_quant=Int4WeightPerTensorFixedPoint,
            bias_quant=Int8BiasPerTensorFixedPointInternalScaling,
            return_quant_tensor=False,
        )

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(f"Expected input shape [N, C, L], but got {x.shape}")
        if x.shape[1] != 1:
            raise ValueError(f"Expected C dimension = 1 for ECG5000 input, but got {x.shape}")
        if x.shape[2] != self.seq_len:
            raise ValueError(
                f"Expected sequence length L = {self.seq_len}, but got {x.shape[2]}"
            )

        x = self.input_quant(x)
        x = self.backbone(x)              # QuantTensor, value shape [N, C_last, L]

        # Flatten on the raw tensor. QuantLinear has input_quant, so the flattened
        # tensor is quantized again before the linear layer.
        x = qt_value(x).contiguous()
        x = self.flatten(x)               # Tensor, shape [N, C_last * L]

        x = self.classifier(x)            # Tensor, shape [N, num_classes]
        return x


if __name__ == "__main__":
    # Quick shape smoke test
    model = ECG5000FullQuantTCN()
    model.eval()

    dummy = torch.randn(1, 1, 140)
    with torch.no_grad():
        y = model(dummy)

    print("input shape :", tuple(dummy.shape))
    print("output shape:", tuple(y.shape))
