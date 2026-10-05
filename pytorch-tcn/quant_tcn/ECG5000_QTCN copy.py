import torch
import torch.nn as nn
import brevitas.nn as qnn

from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias

from .quant_tcn_test import QTCN

# 直接复用你已经写好的 QTCN
# from .quant_tcn import QTCN


class ECG5000QTCNClassifier(nn.Module):
    """
    第二版：
    QTCN backbone + global pooling + quantized classifier

    输入:
        x: [N, L] 或 [N, 1, L]
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
        causal=False,
        use_skip_connections=False,
        input_shape='NCL',
    ):
        super().__init__()

        self.num_classes = num_classes
        self.seq_len = seq_len
        self.input_shape = input_shape

        # 量化输入
        self.quant_in = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=False,
        )

        # 量化 TCN 主干
        self.backbone = QTCN(
            num_inputs=num_inputs,
            num_channels=list(num_channels),
            kernel_size=kernel_size,
            dilations=None,
            dilation_reset=None,
            dropout=dropout,
            causal=causal,
            use_norm=None,
            activation='relu',
            kernel_initializer='xavier_uniform',
            use_skip_connections=use_skip_connections,
            input_shape=input_shape,
            embedding_shapes=None,
            embedding_mode='add',
            use_gate=False,
            lookahead=0,
            output_projection=None,
            output_activation=None,
        )

        # 全局池化，把 [N, C, L] -> [N, C, 1]
        #self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.global_pool = nn.AvgPool1d(kernel_size=seq_len)

        # 池化后再做一次量化 identity，方便分类头前接口更清晰
        self.quant_head_in = qnn.QuantIdentity(
            act_quant=Int8ActPerTensorFloat,
            return_quant_tensor=True,
        )

        # 量化分类头
        # self.classifier = qnn.QuantLinear(
        #     in_features=num_channels[-1],
        #     out_features=num_classes,
        #     bias=True,
        #     weight_quant=Int8WeightPerTensorFloat,
        #     bias_quant=Int32Bias,
        #     return_quant_tensor=False,
        # )

        self.classifier = qnn.QuantLinear(
            in_features=num_channels[-1],
            out_features=num_classes,
            bias=True,
            #input_quant=Int8ActPerTensorFloat,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            return_quant_tensor=False,
        )

    def forward(self, x):
        """
        x:
            [N, L] 或 [N, 1, L]
        """
        # 如果输入是 [N, L]，补成 [N, 1, L]
        if x.ndim == 2:
            x = x.unsqueeze(1)

        if x.ndim != 3:
            raise ValueError(
                f"Expected input with shape [N, L] or [N, C, L], but got {tuple(x.shape)}"
            )

        # 对于 ECG5000，推荐统一使用 NCL: [N, 1, L]
        if self.input_shape == 'NLC':
            # 若你后面真想用 NLC，则这里应传入 [N, L, C]
            # 当前默认 ECG5000 更推荐 NCL
            pass

        x = self.quant_in(x)

        # backbone 输出: [N, C_out, L]
        x = self.backbone(x)

        # 池化: [N, C_out, 1]
        x = self.global_pool(x)

        # 压掉最后一个维度: [N, C_out]
        x = x.squeeze(-1)

        x = self.quant_head_in(x)

        # logits: [N, num_classes]
        x = self.classifier(x)
        return x
