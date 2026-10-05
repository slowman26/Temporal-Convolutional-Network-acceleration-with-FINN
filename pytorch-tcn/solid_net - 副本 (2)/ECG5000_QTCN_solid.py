import torch
import torch.nn as nn
import brevitas.nn as qnn

from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias

from .quant_tcn_solid import TCN


class ECG5000QTCNClassifier(nn.Module):
    """
    QTCN backbone + global pooling + quantized classifier

    推荐输入:
        x: [N, 1, L]

    输出:
        logits: [N, num_classes]

    若传入 in_buffers:
        logits, out_buffers
    """

    def __init__(
        self,
        num_classes=5,
        seq_len=140,
        num_inputs=1,
        num_channels=(16, 16, 32),
        kernel_size=3,
        dropout=0.1,
        causal=True,
        use_skip_connections=False,
        input_shape="NCL",
    ):
        super().__init__()

        self.num_classes = num_classes
        self.seq_len = seq_len
        self.input_shape = input_shape
        self.num_inputs = num_inputs
        self.num_channels = num_channels
        self.causal = causal

        # 输入量化
        self.quant_in = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=False,
        )

        # TCN 主干
        # 输入:  [N, C, L] 或 [N, L, C]（由 input_shape 决定）
        # 输出:  [N, C_out, 1, L]
        self.backbone = TCN(
            num_inputs=num_inputs,
            num_channels=list(num_channels),
            kernel_size=kernel_size,
            dilations=None,
            dilation_reset=None,
            dropout=dropout,
            causal=causal,
            use_norm=None,
            activation="relu",
            kernel_initializer="xavier_uniform",
            use_skip_connections=use_skip_connections,
            input_shape=input_shape,
            embedding_shapes=None,
            embedding_mode="add",
            use_gate=False,
            lookahead=0,
            output_projection=None,
            output_activation=None,
        )

        # 4D 全局池化:
        # [N, C_out, 1, L] -> [N, C_out, 1, 1]
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # [N, C_out, 1, 1] -> [N, C_out]
        self.flatten = nn.Flatten(start_dim=1)

        # 分类头前量化
        self.quant_head_in = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # 量化分类头
        self.classifier = qnn.QuantLinear(
            in_features=num_channels[-1],
            out_features=num_classes,
            bias=True,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            return_quant_tensor=False,
        )

    def make_init_buffers(
        self,
        batch_size=1,
        device=None,
        dtype=None,
    ):
        return self.backbone.make_init_buffers(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def reset_buffers(
        self,
        batch_size=1,
        device=None,
        dtype=None,
    ):
        return self.make_init_buffers(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

    def _format_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        固定要求输入为 3D:
            - NCL: [N, C, L]
            - NLC: [N, L, C]

        对你现在的 ECG5000 训练脚本，通常是 [N, 1, 140]
        """
        if x.ndim != 3:
            raise ValueError(
                f"Expected input with shape [N, C, L] or [N, L, C], but got {tuple(x.shape)}"
            )

        if self.input_shape == "NCL":
            if x.shape[1] != self.num_inputs:
                raise ValueError(
                    f"For input_shape='NCL', expected x.shape[1] == {self.num_inputs}, but got {x.shape[1]}"
                )
        elif self.input_shape == "NLC":
            if x.shape[2] != self.num_inputs:
                raise ValueError(
                    f"For input_shape='NLC', expected x.shape[2] == {self.num_inputs}, but got {x.shape[2]}"
                )
        else:
            raise ValueError(f"Unsupported input_shape: {self.input_shape}")

        return x

    def forward(
        self,
        x: torch.Tensor,
        in_buffers=None,
        return_buffers: bool = False,
    ):
        # 输入保持 3D，进入 backbone 后再统一扩成 4D
        x = self._format_input(x)

        # [N, C, L] or [N, L, C]
        x = self.quant_in(x)

        # backbone 输出: [N, C_out, 1, L]
        x, out_buffers = self.backbone(x, in_buffers=in_buffers)

        # [N, C_out, 1, 1]
        x = self.global_pool(x)

        # [N, C_out]
        x = self.flatten(x)

        x = self.quant_head_in(x)

        # [N, num_classes]
        logits = self.classifier(x)

        if return_buffers or (in_buffers is not None):
            return logits, out_buffers
        return logits