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
from brevitas.quant.fixed_point import (
    Int4WeightPerTensorFixedPointDecoupled,
    
    Int8ActPerTensorFixedPoint,
    Uint8ActPerTensorFixedPoint,
    Int8BiasPerTensorFixedPointInternalScaling,
)

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
    return x.value if hasattr(x, "value") else x


def qt_set(x, value):
    if hasattr(x, "set"):
        return x.set(value=value)
    raise TypeError("Expected QuantTensor input.")

    
class CausalPad2dExport(nn.Module):
    """
    现在时间维在 H 维:
        x: [N, C, L, 1]
    所以 pad 要加在 top/bottom 上，而不是 left/right。
    """
    def __init__(self, pad_left, pad_right=0):
        super().__init__()
        self.pad_left = pad_left
        self.pad_right = pad_right

    def forward(self, x):
        # F.pad 的 4D pad 顺序是:
        # (left, right, top, bottom)
        y = F.pad(qt_value(x), (0, 0, self.pad_left, self.pad_right), "constant", 0.0)
        return qt_set(x, y)

class QuantTemporalConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation=1,
        bias=True,
        causal=True,
    ):
        super().__init__()

        # self.pad_quant = qnn.QuantIdentity(
        #     act_quant=Int4ActPerTensorFixedPoint,
        #     return_quant_tensor=True,
        # )
        

        if causal:
            self.pad_left = (kernel_size - 1) * dilation
            self.pad_right = 0
        else:
            total_pad = (kernel_size - 1) * dilation
            self.pad_left = total_pad // 2
            self.pad_right = total_pad - self.pad_left

        self.pad = CausalPad2dExport(self.pad_left, self.pad_right)

        self.conv = qnn.QuantConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, 1),
            stride=(1, 1),
            dilation=(dilation, 1),
            bias=bias,
            weight_quant=Int4WeightPerTensorFixedPoint,
            bias_quant=Int8BiasPerTensorFixedPointInternalScaling,
            return_quant_tensor=True,
        )

    def forward(self, x):
        # x: [N, C, 1, L]
        x = self.pad(x)
       # x = self.pad_quant(x)
        x = self.conv(x)
        return x

# class QuantTemporalConv2d(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         out_channels,
#         kernel_size,
#         dilation=1,
#         bias=True,
#         causal=True,
#     ):
#         super().__init__()

#         self.pad_quant = qnn.QuantIdentity(
#             act_quant=Int8ActPerTensorFixedPoint,
#             return_quant_tensor=True,
#         )

#         if causal:
#             self.pad_left = (kernel_size - 1) * dilation
#             self.pad_right = 0
#         else:
#             total_pad = (kernel_size - 1) * dilation
#             self.pad_left = total_pad // 2
#             self.pad_right = total_pad - self.pad_left

#         self.conv = qnn.QuantConv2d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=(1, kernel_size),
#             stride=(1, 1),
#             dilation=(1, dilation),
#             bias=bias,
#             weight_quant=Int8WeightPerTensorFixedPoint,
#             bias_quant=Int8BiasPerTensorFixedPointInternalScaling,
#             return_quant_tensor=True,
#         )

#     def forward(self, x):
#         # x: [N, C, 1, L]
#         x = qt_set(
#             x,
#             F.pad(
#                 qt_value(x),
#                 (self.pad_left, self.pad_right, 0, 0),
#                 mode="constant",
#                 value=0.0,
#             ),
#         )
#         x = self.pad_quant(x)
#         x = self.conv(x)
#         return x


class QuantTemporalBlock(nn.Module):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        dilation,
        dropout=0.0,
        causal=True,
    ):
        super().__init__()

        self.drop1_p = dropout
        self.drop2_p = dropout

        self.conv1 = QuantTemporalConv2d(
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

        self.conv2 = QuantTemporalConv2d(
            in_channels=n_outputs,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            dilation=dilation,
            bias=True,
            causal=causal,
        )

        

        if n_inputs != n_outputs:
            self.downsample = qnn.QuantConv2d(
                in_channels=n_inputs,
                out_channels=n_outputs,
                kernel_size=(1, 1),
                stride=(1, 1),
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

        out = self.pre_add_quant(qt_value(out))
        res = self.pre_add_quant(qt_value(res))

        # print("out scale:", out.scale)
        # print("res scale:", res.scale)
        # print("out bit_width:", out.bit_width)
        # print("res bit_width:", res.bit_width)

        # out = qt_value(out)
        # res = qt_value(res)

        out = out + res
        out = self.post_add_act(out)
        return out


class ECG5000FullQuantTCN(nn.Module):
    def __init__(
        self,
        num_classes=5,
        seq_len=140,
        num_inputs=1,
        num_channels=(16, 16, 16),
        kernel_size=5,
        dropout=0.0,
        causal=True,
        #export_all_time_logits=False,
        
    ):
        super().__init__()

        self.seq_len = seq_len
        #self.export_all_time_logits=export_all_time_logits

        self.input_quant = qnn.QuantIdentity(
            act_quant=Int4ActPerTensorFixedPoint,
            return_quant_tensor=True,
        )
        # self.fc = qnn.QuantLinear(
        #     in_features=num_channels[-1] * seq_len,
        #     out_features=num_classes,
        #     bias=True,
        #     input_quant=Int8ActPerTensorFixedPoint,
        #     weight_quant=Int8WeightPerTensorFixedPoint,
        #     bias_quant=Int8BiasPerTensorFixedPointInternalScaling,
        #     return_quant_tensor=False,
        # )

        # self.classifier = qnn.QuantConv2d(
        #     in_channels=num_channels[-1],
        #     out_channels=num_classes,
        #     kernel_size=(seq_len, 1),
        #     stride=(1, 1),
        #     bias=True,
        #     weight_quant=Int8WeightPerTensorFixedPoint,
        #     bias_quant=Int8BiasPerTensorFixedPointInternalScaling,
        #     return_quant_tensor=True,
            
        # )

        self.classifier = qnn.QuantConv2d(
            in_channels=num_channels[-1],
            out_channels=num_classes,
            kernel_size=(seq_len, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
            weight_quant=Int4WeightPerTensorFixedPoint,
            bias_quant=Int8BiasPerTensorFixedPointInternalScaling,
            return_quant_tensor=False,
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
                    #dilation=2 ** i,
                    dilation=1,
                    dropout=dropout,
                    causal=causal,
                    
                )
            )
            in_ch = out_ch

        self.backbone = nn.Sequential(*layers)

        # self.fc = qnn.QuantLinear(
        #     in_features=num_channels[-1] * seq_len,
        #     out_features=num_classes,
        #     bias=True,
        #     weight_quant=Int8WeightPerTensorFixedPoint,
        #     bias_quant=Int8BiasPerTensorFixedPointInternalScaling,
        #     return_quant_tensor=False,
        # )

    def forward(self, x):
        if x.dim() != 4:
            raise ValueError(f"Expected input shape [N, C, L, 1], but got {x.shape}")
        if x.shape[3] != 1:
            raise ValueError(f"Expected W dimension = 1, but got {x.shape}")

        #x = ExportWrapper(x)
        x = self.input_quant(x)
        x = self.backbone(x)

        x = self.classifier(x)
        #x = qt_value(x)

        # if self.export_all_time_logits:
        #     # Shape: [N, 5, L, 1]
        #     # FINN can keep this as pure Conv/MVAU output.
        #     return x    
        
        return x
    

# class ECG5000FullQuantTCN(nn.Module):
#     def __init__(
#         self,
#         num_classes=5,
#         seq_len=140,
#         num_inputs=1,
#         num_channels=(16, 16, 16),
#         kernel_size=5,
#         dropout=0.0,
#         causal=True,
#     ):
#         super().__init__()

#         self.seq_len = seq_len

#         self.input_quant = qnn.QuantIdentity(
#             act_quant=Int4ActPerTensorFixedPoint,
#             return_quant_tensor=True,
#         )

#         layers = []
#         in_ch = num_inputs
#         for i, out_ch in enumerate(num_channels):
#             print("in_ch =", in_ch)
#             print("out_ch =", out_ch)

#             layers.append(
#                 QuantTemporalBlock(
#                     n_inputs=in_ch,
#                     n_outputs=out_ch,
#                     kernel_size=kernel_size,
#                     dilation=2 ** i,
#                     dropout=dropout,
#                     causal=causal,
#                 )
#             )
#             in_ch = out_ch

#         self.backbone = nn.Sequential(*layers)

#         # -----------------------------------------------------
#         # Flatten + Linear classifier
#         # Input before flatten: [N, C, L, 1]
#         # After flatten:        [N, C * L]
#         # Here C = num_channels[-1], L = seq_len
#         # -----------------------------------------------------
#         self.flatten = nn.Flatten(start_dim=1)

#         self.classifier = qnn.QuantLinear(
#             in_features=num_channels[-1] * seq_len,
#             out_features=num_classes,
#             bias=True,
#             input_quant=Int4ActPerTensorFixedPoint,
#             weight_quant=Int4WeightPerTensorFixedPoint,
#             bias_quant=Int8BiasPerTensorFixedPointInternalScaling,
#             return_quant_tensor=True,
#         )

#     def forward(self, x):
#         if x.dim() != 4:
#             raise ValueError(f"Expected input shape [N, C, L, 1], but got {x.shape}")
#         if x.shape[3] != 1:
#             raise ValueError(f"Expected W dimension = 1, but got {x.shape}")

#         x = self.input_quant(x)
#         x = self.backbone(x)

#         x = self.flatten(x)
#         x = self.classifier(x)

#         return x


    
    # def forward(self, x):
    #     if x.dim() != 4:
    #         raise ValueError(f"Expected input shape [N, C, 1, L], but got {x.shape}")
    #     if x.shape[2] != 1:
    #         raise ValueError(f"Expected H dimension = 1, but got {x.shape}")
    
    #     x = self.input_quant(x)
    #     x = self.backbone(x)
    
    #     # 到这里 x 还是 QuantTensor
    #     x = qt_value(x).contiguous().view(x.shape[0], -1)
    
    #     print("FINAL FC INPUT SHAPE:", x.shape)
    #     print("FC IN_FEATURES:", self.fc.in_features)
    
    #     assert x.dim() == 2
    #     assert x.shape[1] == self.fc.in_features
        
    #     return self.fc(x)