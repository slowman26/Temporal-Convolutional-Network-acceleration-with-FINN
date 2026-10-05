import warnings
import torch
import torch.nn as nn
from numpy.typing import ArrayLike
import numpy as np

import brevitas.nn as qnn
from brevitas.quant import (
    Int8WeightPerTensorFloat,
    Int8ActPerTensorFloat,
    Uint8ActPerTensorFloat,
    Int32Bias,
)

try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm
    warnings.warn(
        "The deprecated weight_norm from torch.nn.utils.weight_norm was imported."
    )

from typing import Tuple, Optional

from solid_net.quant_conv_solid import TemporalConv2dExport, PointwiseConv2dExport


activation_fn = dict(
    relu=nn.ReLU,
    tanh=nn.Tanh,
    leaky_relu=nn.LeakyReLU,
    sigmoid=nn.Sigmoid,
    elu=nn.ELU,
    gelu=nn.GELU,
    selu=nn.SELU,
    softmax=nn.Softmax,
    log_softmax=nn.LogSoftmax,
)

kernel_init_fn = dict(
    xavier_uniform=nn.init.xavier_uniform_,
    xavier_normal=nn.init.xavier_normal_,
    kaiming_uniform=nn.init.kaiming_uniform_,
    kaiming_normal=nn.init.kaiming_normal_,
    normal=nn.init.normal_,
    uniform=nn.init.uniform_,
)


def _check_activation_arg(activation, arg_name):
    if activation is None and arg_name == "output_activation":
        return
    if isinstance(activation, str):
        if activation not in activation_fn.keys():
            raise ValueError(
                f"If argument '{arg_name}' is a string, it must be one of: {activation_fn.keys()}."
            )
    else:
        try:
            if not isinstance(activation(), nn.Module):
                raise ValueError(
                    f"The argument '{arg_name}' must either be a valid string or a torch.nn.Module object."
                )
        except Exception:
            raise ValueError(
                f"The argument '{arg_name}' must either be a valid string or a torch.nn.Module object."
            )


def _check_generic_input_arg(arg, arg_name, allowed_values):
    if arg not in allowed_values:
        raise ValueError(
            f"Argument '{arg_name}' must be one of: {allowed_values}, but {arg} was passed."
        )


def get_kernel_init_fn(name: str, activation: str) -> Tuple[nn.Module, dict]:
    try:
        if isinstance(activation(), nn.Module):
            return kernel_init_fn[name], dict()
    except Exception:
        pass

    if name not in kernel_init_fn.keys():
        raise ValueError(
            f"Argument 'kernel_initializer' must be one of: {kernel_init_fn.keys()}"
        )

    if name in ["xavier_uniform", "xavier_normal"]:
        if activation in ["gelu", "elu", "softmax", "log_softmax"]:
            warnings.warn(
                f"kernel_initializer={name} is not fully compatible with activation={activation}; using gain=sqrt(2)."
            )
            gain = np.sqrt(2)
        else:
            gain = nn.init.calculate_gain(activation)
        kernel_init_kw = dict(gain=gain)

    elif name in ["kaiming_uniform", "kaiming_normal"]:
        if activation in ["gelu", "elu", "softmax", "log_softmax"]:
            raise ValueError(
                f"kernel_initializer={name} is not compatible with activation={activation}."
            )
        else:
            nonlinearity = activation
        kernel_init_kw = dict(nonlinearity=nonlinearity)

    else:
        kernel_init_kw = dict()

    return kernel_init_fn[name], kernel_init_kw


def make_quant_act():
    # ReLU 后面必须用 unsigned activation quant
    return qnn.QuantReLU(
        act_quant=Uint8ActPerTensorFloat,
        return_quant_tensor=False,
    )


def make_quant_pointwise(in_ch, out_ch, bias=True):
    return PointwiseConv2dExport(
        in_channels=in_ch,
        out_channels=out_ch,
        bias=bias,
        input_quant=Int8ActPerTensorFloat,
        weight_quant=Int8WeightPerTensorFloat,
        bias_quant=Int32Bias if bias else None,
        return_quant_tensor=False,
    )


def make_activation_module(activation):
    if isinstance(activation, str):
        if activation == "relu":
            return make_quant_act()
        return activation_fn[activation]()
    return activation()


class BaseTCN(nn.Module):
    def __init__(self):
        super().__init__()

    def inference(self, *args, **kwargs):
        return self(*args, **kwargs)

    def make_init_buffers(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        # 这版是静态 4D 版，不使用 streaming buffer
        return []

    def reset_buffers(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        return []


class TemporalBlock(BaseTCN):
    def __init__(
        self,
        n_inputs,
        n_outputs,
        kernel_size,
        stride,
        dilation,
        dropout,
        causal,
        use_norm,
        activation,
        kerner_initializer,
        embedding_shapes,
        embedding_mode,
        use_gate,
    ):
        super().__init__()

        if not causal:
            raise ValueError("This 4D FINN-oriented version only supports causal=True.")
        if embedding_shapes is not None:
            raise ValueError("embedding_shapes is not supported in this clean 4D version.")

        self.use_norm = use_norm
        self.activation = activation
        self.kernel_initializer = kerner_initializer
        self.embedding_shapes = embedding_shapes
        self.embedding_mode = embedding_mode
        self.use_gate = use_gate
        self.causal = causal
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        conv1_n_outputs = 2 * n_outputs if self.use_gate else n_outputs

        self.conv1 = TemporalConv2dExport(
            in_channels=n_inputs,
            out_channels=conv1_n_outputs,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            input_quant=Int8ActPerTensorFloat,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            return_quant_tensor=False,
        )

        self.conv2 = TemporalConv2dExport(
            in_channels=n_outputs,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            input_quant=Int8ActPerTensorFloat,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            return_quant_tensor=False,
        )

        if use_norm == "batch_norm":
            self.norm1 = nn.BatchNorm2d(conv1_n_outputs)
            self.norm2 = nn.BatchNorm2d(n_outputs)
        elif use_norm == "layer_norm":
            self.norm1 = nn.LayerNorm(conv1_n_outputs)
            self.norm2 = nn.LayerNorm(n_outputs)
        elif use_norm in ["weight_norm", None]:
            self.norm1 = None
            self.norm2 = None
        else:
            raise ValueError(f"Unsupported use_norm={use_norm}")

        if self.use_gate:
            self.activation1 = nn.GLU(dim=1)
        else:
            self.activation1 = make_activation_module(self.activation)

        self.activation2 = make_activation_module(self.activation)
        self.activation_final = make_activation_module(self.activation)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = (
            make_quant_pointwise(n_inputs, n_outputs, bias=True)
            if n_inputs != n_outputs
            else None
        )

        self.init_weights()

        if use_norm == "weight_norm":
            self.conv1.conv = weight_norm(self.conv1.conv)
            self.conv2.conv = weight_norm(self.conv2.conv)

    def init_weights(self):
        initialize, kwargs = get_kernel_init_fn(
            name=self.kernel_initializer,
            activation=self.activation,
        )

        initialize(self.conv1.conv.weight, **kwargs)
        initialize(self.conv2.conv.weight, **kwargs)

        if self.downsample is not None:
            initialize(self.downsample.conv.weight, **kwargs)

    def apply_norm(self, norm_fn, x):
        if norm_fn is None:
            return x

        if self.use_norm == "batch_norm":
            return norm_fn(x)

        if self.use_norm == "layer_norm":
            # [N, C, 1, L] -> [N, 1, L, C]，沿 channel 做 LN
            x = x.permute(0, 2, 3, 1)
            x = norm_fn(x)
            x = x.permute(0, 3, 1, 2)
            return x

        return x

    def forward(
        self,
        x,
        embeddings=None,
        in_buffers=None,
    ):
        if embeddings is not None:
            raise ValueError("embeddings is not supported in this clean 4D version.")

        out = self.conv1(x)
        out = self.apply_norm(self.norm1, out)
        out = self.activation1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.apply_norm(self.norm2, out)
        out = self.activation2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        y = self.activation_final(out + res)

        return y, out, []


class TCN(BaseTCN):
    def __init__(
        self,
        num_inputs: int,
        num_channels: ArrayLike,
        kernel_size: int = 4,
        dilations: Optional[ArrayLike] = None,
        dilation_reset: Optional[int] = None,
        dropout: float = 0.1,
        causal: bool = True,
        use_norm: str = "weight_norm",
        activation: str = "relu",
        kernel_initializer: str = "xavier_uniform",
        use_skip_connections: bool = False,
        input_shape: str = "NCL",
        embedding_shapes: Optional[ArrayLike] = None,
        embedding_mode: str = "add",
        use_gate: bool = False,
        lookahead=0,
        output_projection: Optional[int] = None,
        output_activation: Optional[str] = None,
    ):
        super().__init__()

        if not causal:
            raise ValueError("This 4D FINN-oriented version only supports causal=True.")
        if lookahead > 0:
            raise ValueError("lookahead must be 0.")
        if input_shape != "NCL":
            raise ValueError("This clean 4D version only supports input_shape='NCL'.")
        if embedding_shapes is not None:
            raise ValueError("embedding_shapes is not supported in this clean 4D version.")

        if dilations is not None and len(dilations) != len(num_channels):
            raise ValueError("Length of dilations must match length of num_channels")

        self.allowed_norm_values = ["batch_norm", "layer_norm", "weight_norm", None]
        _check_generic_input_arg(use_norm, "use_norm", self.allowed_norm_values)
        _check_activation_arg(activation, "activation")
        _check_generic_input_arg(
            kernel_initializer, "kernel_initializer", kernel_init_fn.keys()
        )
        _check_generic_input_arg(
            use_skip_connections, "use_skip_connections", [True, False]
        )
        _check_generic_input_arg(use_gate, "use_gate", [True, False])
        _check_activation_arg(output_activation, "output_activation")

        if dilations is None:
            if dilation_reset is None:
                dilations = [2 ** i for i in range(len(num_channels))]
            else:
                dilation_reset = int(np.log2(dilation_reset * 2))
                dilations = [2 ** (i % dilation_reset) for i in range(len(num_channels))]

        self.dilations = dilations
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.use_skip_connections = use_skip_connections
        self.input_shape = input_shape
        self.use_gate = use_gate
        self.causal = causal
        self.output_projection = output_projection
        self.output_activation = output_activation

        if use_skip_connections:
            self.downsample_skip_connection = nn.ModuleList()
            for i in range(len(num_channels)):
                if num_channels[i] != num_channels[-1]:
                    self.downsample_skip_connection.append(
                        PointwiseConv2dExport(
                            in_channels=num_channels[i],
                            out_channels=num_channels[-1],
                            bias=True,
                            input_quant=Int8ActPerTensorFloat,
                            weight_quant=Int8WeightPerTensorFloat,
                            bias_quant=Int32Bias,
                            return_quant_tensor=False,
                        )
                    )
                else:
                    self.downsample_skip_connection.append(None)

            self.init_skip_connection_weights()
            self.activation_skip_out = make_activation_module(self.activation)
        else:
            self.downsample_skip_connection = None

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = self.dilations[i]
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    dropout=dropout,
                    causal=causal,
                    use_norm=use_norm,
                    activation=activation,
                    kerner_initializer=self.kernel_initializer,
                    embedding_shapes=None,
                    embedding_mode="add",
                    use_gate=self.use_gate,
                )
            )

        self.network = nn.ModuleList(layers)

        if self.output_projection is not None:
            self.projection_out = PointwiseConv2dExport(
                in_channels=num_channels[-1],
                out_channels=self.output_projection,
                bias=True,
                input_quant=Int8ActPerTensorFloat,
                weight_quant=Int8WeightPerTensorFloat,
                bias_quant=Int32Bias,
                return_quant_tensor=False,
            )
        else:
            self.projection_out = None

        if self.output_activation is not None:
            self.activation_out = make_activation_module(self.output_activation)
        else:
            self.activation_out = None

    def init_skip_connection_weights(self):
        initialize, kwargs = get_kernel_init_fn(
            name=self.kernel_initializer,
            activation=self.activation,
        )
        for layer in self.downsample_skip_connection:
            if layer is not None:
                initialize(layer.conv.weight, **kwargs)

    def forward(
        self,
        x,
        embeddings=None,
        in_buffers=None,
    ):
        if x.ndim != 4:
            raise ValueError(f"TCN expects 4D input [N, C, 1, L], but got {tuple(x.shape)}")
        if x.shape[2] != 1:
            raise ValueError(f"TCN expects height dimension == 1, but got {x.shape[2]}")
        if embeddings is not None:
            raise ValueError("embeddings is not supported in this clean 4D version.")

        out_buffers = []

        if self.use_skip_connections:
            skip_connections = []

            for index, layer in enumerate(self.network):
                x, skip_out, layer_out_buffers = layer(
                    x,
                    embeddings=None,
                    in_buffers=None,
                )
                out_buffers.extend(layer_out_buffers)

                if self.downsample_skip_connection[index] is not None:
                    skip_out = self.downsample_skip_connection[index](skip_out)

                if index < len(self.network) - 1:
                    skip_connections.append(skip_out)

            skip_connections.append(x)
            x = torch.stack(skip_connections, dim=0).sum(dim=0)
            x = self.activation_skip_out(x)
        else:
            for layer in self.network:
                x, _, layer_out_buffers = layer(
                    x,
                    embeddings=None,
                    in_buffers=None,
                )
                out_buffers.extend(layer_out_buffers)

        if self.projection_out is not None:
            x = self.projection_out(x)

        if self.activation_out is not None:
            x = self.activation_out(x)

        return x, out_buffers