import torch
import torch.nn as nn
import brevitas.nn as qnn

from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias

from .quant_tcn_solid import TCN


class ECG5000QTCNClassifier(nn.Module):
    """
    4D-internal QTCN backbone + global pooling + quantized classifier

    期望输入:
        x: [N, 1, L]

    backbone 内部:
        [N, C, 1, L]

    输出:
        logits: [N, num_classes]
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

        if input_shape != "NCL":
            raise ValueError("This classifier version only supports input_shape='NCL'.")

        self.num_classes = num_classes
        self.seq_len = seq_len
        self.input_shape = input_shape
        self.num_inputs = num_inputs
        self.num_channels = num_channels
        self.causal = causal

        # 输入量化：Identity 类型，保持 signed
        self.quant_in = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=False,
        )

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

        # [N, C, 1, L] -> [N, C, 1, 1]
        self.global_pool = nn.AvgPool2d(kernel_size=(1, seq_len))

        # 用 Flatten 代替 squeeze，避免再生成 If(Squeeze, Identity)
        self.flatten = nn.Flatten(start_dim=1)

        # head 前的 Identity 量化保持 signed
        self.quant_head_in = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

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
        return []

    def reset_buffers(
        self,
        batch_size=1,
        device=None,
        dtype=None,
    ):
        return []

    def _format_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected input with shape [N, C, L], but got {tuple(x.shape)}"
            )
        if x.shape[1] != self.num_inputs:
            raise ValueError(
                f"Expected x.shape[1] == {self.num_inputs}, but got {x.shape[1]}"
            )
        return x

    def forward(
        self,
        x: torch.Tensor,
        in_buffers=None,
        return_buffers: bool = False,
    ):
        x = self._format_input(x)

        # [N, 1, L]
        x = self.quant_in(x)

        # 只在模型入口做一次 4D 扩展
        x = x.unsqueeze(2)  # [N, 1, 1, L]

        # backbone 输出: [N, C_out, 1, L]
        x, out_buffers = self.backbone(x, in_buffers=None)

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