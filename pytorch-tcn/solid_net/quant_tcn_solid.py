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
from collections.abc import Iterable

from solid_net.quant_conv_solid import TemporalConv1dExport, PointwiseConv1dExport
from solid_net.quant_pad_solid import TemporalPad2dTrain


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
    return


def _check_generic_input_arg(arg, arg_name, allowed_values):
    if arg not in allowed_values:
        raise ValueError(
            f"Argument '{arg_name}' must be one of: {allowed_values}, but {arg} was passed."
        )
    return


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
    return qnn.QuantReLU(
        act_quant=Uint8ActPerTensorFloat,
        return_quant_tensor=True,
    )


def make_quant_pointwise(in_ch, out_ch, bias=False):
    return PointwiseConv1dExport(
        in_channels=in_ch,
        out_channels=out_ch,
        bias=bias,
        input_quant=Int8ActPerTensorFloat,
        weight_quant=Int8WeightPerTensorFloat,
        bias_quant=Int32Bias if bias else None,
        return_quant_tensor=True,
    )


def make_activation_module(activation):
    if isinstance(activation, str):
        if activation == "relu":
            return make_quant_act()
        return activation_fn[activation]()
    else:
        return activation()


class ChannelLayerNorm4d(nn.Module):
    """
    对 4D 张量 [B, C, 1, T] 在 channel 维做 LayerNorm。
    注意：这会引入 permute，不建议用于 FINN 导出主路径。
    更推荐 use_norm='weight_norm' 或 None。
    """
    def __init__(self, num_channels: int):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)   # [B, C, 1, T] -> [B, 1, T, C]
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)   # [B, 1, T, C] -> [B, C, 1, T]
        return x


class BaseTCN(nn.Module):
    def __init__(self):
        super(BaseTCN, self).__init__()

    def inference(self, *args, **kwargs):
        return self(*args, **kwargs)

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
                m.weight.data.normal_(0.0, 0.01)

        self.apply(_init_weights)

    def make_init_buffers(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        raise NotImplementedError

    def reset_buffers(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        return self.make_init_buffers(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

class ChannelTransitionBlock(BaseTCN):
    """
    单路径通道变换块：
    [B, C_in, 1, T] -> [B, C_out, 1, T]

    用 1x1 pointwise conv 做通道调整，不做 residual add。
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        activation,
        kernel_initializer,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_name = activation
        self.kernel_initializer = kernel_initializer

        self.proj = make_quant_pointwise(
            in_ch=in_channels,
            out_ch=out_channels,
            bias=False,
        )
        self.activation = make_activation_module(activation)

        self.init_weights()

    def init_weights(self):
        initialize, kwargs = get_kernel_init_fn(
            name=self.kernel_initializer,
            activation=self.activation_name,
        )
        initialize(self.proj.conv.weight, **kwargs)

    def make_init_buffers(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        return []

    def forward(
        self,
        x,
        embeddings=None,
        in_buffers=None,
    ):
        y = self.proj(x)
        y = self.activation(y)
        return y, y, []


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


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
        weight_bit_width=8,
        act_bit_width=8,
        padder_cls=TemporalPad2dTrain,
    ):
        super(TemporalBlock, self).__init__()

        if not causal:
            raise ValueError(
                "This fixed-buffer FINN-friendly version only supports causal=True."
            )

        if n_inputs != n_outputs:
            raise ValueError(
                f"TemporalBlock now only supports same-channel residual blocks, "
                f"but got n_inputs={n_inputs}, n_outputs={n_outputs}. "
                f"Use ChannelTransitionBlock before this block."
            )

        self.use_norm = use_norm
        self.activation = activation
        self.kernel_initializer = kerner_initializer
        self.embedding_shapes = embedding_shapes
        self.embedding_mode = embedding_mode
        self.use_gate = use_gate
        self.causal = causal
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        conv1d_n_outputs = 2 * n_outputs if self.use_gate else n_outputs

        self.conv1 = TemporalConv1dExport(
            in_channels=n_inputs,
            out_channels=conv1d_n_outputs,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=False,
            input_quant=Int8ActPerTensorFloat,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=True,
            padder_cls=padder_cls,
        )

        self.conv2 = TemporalConv1dExport(
            in_channels=n_outputs,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            bias=False,
            input_quant=Int8ActPerTensorFloat,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=None,
            return_quant_tensor=True,
            padder_cls=padder_cls,
        )

        if use_norm == "batch_norm":
            self.norm1 = nn.BatchNorm2d(conv1d_n_outputs)
            self.norm2 = nn.BatchNorm2d(n_outputs)
        elif use_norm == "layer_norm":
            self.norm1 = ChannelLayerNorm4d(conv1d_n_outputs)
            self.norm2 = ChannelLayerNorm4d(n_outputs)
        elif use_norm == "weight_norm":
            self.norm1 = None
            self.norm2 = None
        elif use_norm is None:
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

        if self.embedding_shapes is not None:
            embedding_layer_n_outputs = conv1d_n_outputs

            self.embedding_projection_1 = nn.Conv2d(
                in_channels=sum([shape[0] for shape in self.embedding_shapes]),
                out_channels=embedding_layer_n_outputs,
                kernel_size=(1, 1),
                bias=True,
            )

            self.embedding_projection_2 = nn.Conv2d(
                in_channels=2 * embedding_layer_n_outputs,
                out_channels=embedding_layer_n_outputs,
                kernel_size=(1, 1),
                bias=True,
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

        if self.embedding_shapes is not None:
            initialize(self.embedding_projection_1.weight, **kwargs)
            initialize(self.embedding_projection_2.weight, **kwargs)

    def make_init_buffers(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        return []

    def apply_norm(self, norm_fn, x):
        if norm_fn is None:
            return x
        return norm_fn(x)

    def apply_embeddings(self, x, embeddings):
        if not isinstance(embeddings, list):
            embeddings = [embeddings]

        e = []
        T = x.shape[3]

        for embedding, expected_shape in zip(embeddings, self.embedding_shapes):
            if embedding.shape[1] != expected_shape[0]:
                raise ValueError(
                    f"Embedding shape {embedding.shape} does not match expected shape {expected_shape}."
                )

            if len(embedding.shape) == 2:
                e.append(embedding.unsqueeze(2).unsqueeze(3).repeat(1, 1, 1, T))
            elif len(embedding.shape) == 3:
                if embedding.shape[2] != T:
                    raise ValueError(
                        f"Embedding time dimension {embedding.shape[2]} does not match input time dimension {T}."
                    )
                e.append(embedding.unsqueeze(2))
            elif len(embedding.shape) == 4:
                if embedding.shape[2] != 1 or embedding.shape[3] != T:
                    raise ValueError(
                        f"Embedding shape {embedding.shape} does not match expected [B, C_emb, 1, {T}]."
                    )
                e.append(embedding)
            else:
                raise ValueError(
                    f"Unsupported embedding ndim={embedding.ndim}, shape={embedding.shape}"
                )

        e = torch.cat(e, dim=1)
        e = self.embedding_projection_1(e)

        if self.embedding_mode == "concat":
            x = self.embedding_projection_2(torch.cat([x, e], dim=1))
        elif self.embedding_mode == "add":
            x = x + e

        return x

    def forward(
        self,
        x,
        embeddings=None,
        in_buffers=None,
    ):
        out = self.conv1(x)
        out = self.apply_norm(self.norm1, out)

        if embeddings is not None:
            out = self.apply_embeddings(out, embeddings)

        out = self.activation1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.apply_norm(self.norm2, out)
        out = self.activation2(out)
        out = self.dropout2(out)

        y = self.activation_final(out + x)

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
        padder_cls=TemporalPad2dTrain,
    ):
        super(TCN, self).__init__()

        if not causal:
            raise ValueError(
                "This fixed-buffer FINN-friendly version only supports causal=True."
            )

        if lookahead > 0:
            raise ValueError("lookahead must be 0.")

        if dilations is not None and len(dilations) != len(num_channels):
            raise ValueError("Length of dilations must match length of num_channels")

        self.allowed_norm_values = ["batch_norm", "layer_norm", "weight_norm", None]
        self.allowed_input_shapes = ["NCL", "NLC"]

        _check_generic_input_arg(causal, "causal", [True, False])
        _check_generic_input_arg(use_norm, "use_norm", self.allowed_norm_values)
        _check_activation_arg(activation, "activation")
        _check_generic_input_arg(
            kernel_initializer, "kernel_initializer", kernel_init_fn.keys()
        )
        _check_generic_input_arg(
            use_skip_connections, "use_skip_connections", [True, False]
        )
        _check_generic_input_arg(input_shape, "input_shape", self.allowed_input_shapes)
        _check_generic_input_arg(embedding_mode, "embedding_mode", ["add", "concat"])
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
        self.embedding_shapes = embedding_shapes
        self.embedding_mode = embedding_mode
        self.use_gate = use_gate
        self.causal = causal
        self.output_projection = output_projection
        self.output_activation = output_activation
        self.padder_cls = padder_cls

        if embedding_shapes is not None:
            if isinstance(embedding_shapes, Iterable):
                for shape in embedding_shapes:
                    if not isinstance(shape, tuple):
                        try:
                            shape = tuple(shape)
                        except Exception as e:
                            raise ValueError(
                                f"Each shape in 'embedding_shapes' must be convertible to tuple. Error: {e}"
                            )
                    if len(shape) not in [1, 2]:
                        raise ValueError(
                            "Tuples in 'embedding_shapes' must be of length 1 or 2."
                        )
            else:
                raise ValueError(
                    f"'embedding_shapes' must be an Iterable of tuples, but got {type(embedding_shapes)}."
                )

        layers = []
        layer_out_channels = []
        current_in_channels = num_inputs
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = self.dilations[i]
            target_out_channels = num_channels[i]

            if current_in_channels != target_out_channels:
                layers.append(
                    ChannelTransitionBlock(
                        in_channels=current_in_channels,
                        out_channels=target_out_channels,
                        activation=activation,
                        kernel_initializer=self.kernel_initializer,
                    )
                )
                layer_out_channels.append(target_out_channels)
                current_in_channels = target_out_channels

            layers.append(
                TemporalBlock(
                    n_inputs=current_in_channels,
                    n_outputs=target_out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    dropout=dropout,
                    causal=causal,
                    use_norm=use_norm,
                    activation=activation,
                    kerner_initializer=self.kernel_initializer,
                    embedding_shapes=self.embedding_shapes,
                    embedding_mode=self.embedding_mode,
                    use_gate=self.use_gate,
                    padder_cls=self.padder_cls,
                )
            )
            layer_out_channels.append(target_out_channels)
            current_in_channels = target_out_channels

        self.network = nn.ModuleList(layers)
        self.layer_out_channels = layer_out_channels

        if use_skip_connections:
            self.downsample_skip_connection = nn.ModuleList()
            final_channels = num_channels[-1]

            for ch in self.layer_out_channels:
                if ch != final_channels:
                    self.downsample_skip_connection.append(
                        make_quant_pointwise(ch, final_channels, bias=False)
                    )
                else:
                    self.downsample_skip_connection.append(None)

            self.init_skip_connection_weights()
            self.activation_skip_out = make_activation_module(self.activation)
        else:
            self.downsample_skip_connection = None

        if self.output_projection is not None:
            self.projection_out = make_quant_pointwise(
                num_channels[-1], self.output_projection, bias=False
            )
        else:
            self.projection_out = None

        if self.output_activation is not None:
            self.activation_out = make_activation_module(self.output_activation)
        else:
            self.activation_out = None

    def make_init_buffers(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        return []

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
        if self.input_shape == "NLC":
            x = x.transpose(1, 2)

        if x.ndim != 3:
            raise ValueError(f"TCN expects 3D input before internal expand, got {tuple(x.shape)}")

        x = x.unsqueeze(2)
        out_buffers = []

        if self.use_skip_connections:
            skip_connections = []

            for index, layer in enumerate(self.network):
                x, skip_out, layer_out_buffers = layer(
                    x,
                    embeddings=embeddings,
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
            for index, layer in enumerate(self.network):
                x, _, layer_out_buffers = layer(
                    x,
                    embeddings=embeddings,
                    in_buffers=None,
                )
                out_buffers.extend(layer_out_buffers)

        if self.projection_out is not None:
            x = self.projection_out(x)

        if self.activation_out is not None:
            x = self.activation_out(x)

        return x, out_buffers