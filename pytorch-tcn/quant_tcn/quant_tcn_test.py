import warnings
import torch
import torch.nn as nn
import numpy as np

import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias

from brevitas.quant import Uint8ActPerTensorFloat

from typing import Tuple
from typing import Optional
from collections.abc import Iterable
from numpy.typing import ArrayLike

from .quant_conv import TemporalConv1d, TemporalConvTranspose1d
from .quant_pad import TemporalPad1d
from .quant_buffer import BufferIO


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
    if activation is None and arg_name == 'output_activation':
        return
    if isinstance(activation, str):
        if activation not in activation_fn.keys():
            raise ValueError(
                f"""
                If argument '{arg_name}' is a string, it must be one of:
                {activation_fn.keys()}.
                """
            )
    else:
        try:
            if not isinstance(activation(), nn.Module):
                raise ValueError(
                    f"""
                    The argument '{arg_name}' must either be a valid string or
                    a torch.nn.Module object, but got {type(activation())}.
                    """
                )
        except Exception:
            raise ValueError(
                f"""
                The argument '{arg_name}' must either be a valid string or
                a torch.nn.Module object, but got {type(activation)}.
                """
            )


def _check_generic_input_arg(arg, arg_name, allowed_values):
    if arg not in allowed_values:
        raise ValueError(
            f"""
            Argument '{arg_name}' must be one of: {allowed_values},
            but {arg} was passed.
            """
        )


def get_kernel_init_fn(name: str, activation: str) -> Tuple[nn.Module, dict]:
    if name not in kernel_init_fn.keys():
        raise ValueError(
            f"Argument 'kernel_initializer' must be one of: {kernel_init_fn.keys()}"
        )

    if name in ['xavier_uniform', 'xavier_normal']:
        if activation in ['gelu', 'elu', 'softmax', 'log_softmax']:
            warnings.warn(
                f"""
                Argument 'kernel_initializer' {name}
                is not fully compatible with activation {activation}.
                Here, a gain of sqrt(2) is used.
                """
            )
            gain = np.sqrt(2)
        else:
            gain = nn.init.calculate_gain(activation)
        kernel_init_kw = dict(gain=gain)

    elif name in ['kaiming_uniform', 'kaiming_normal']:
        if activation in ['gelu', 'elu', 'softmax', 'log_softmax']:
            raise ValueError(
                f"""
                Argument 'kernel_initializer' {name}
                is not compatible with activation {activation}.
                """
            )
        kernel_init_kw = dict(nonlinearity=activation)
    else:
        kernel_init_kw = dict()

    return kernel_init_fn[name], kernel_init_kw


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


def make_quant_act():
    return qnn.QuantReLU(
        act_quant=Uint8ActPerTensorFloat,
        return_quant_tensor=False,
    )


def make_quant_identity():
    return qnn.QuantIdentity(
        act_quant=Int8ActPerTensorFloat,
        return_quant_tensor=False,
    )


def make_quant_pointwise(in_ch, out_ch, bias=True):
    return QuantPointwiseConv1dAs2d(in_ch, out_ch, bias=bias)

class QuantPointwiseConv1dAs2d(nn.Module):
    def __init__(self, in_ch, out_ch, bias=True):
        super().__init__()
        self.conv = qnn.QuantConv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            bias=bias,
            input_quant=Int8ActPerTensorFloat,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias if bias else None,
            return_quant_tensor=False,
        )

    @property
    def weight(self):
        return self.conv.weight

    @property
    def bias(self):
        return self.conv.bias

    def forward(self, x):
        # x: [B, C, 1, T]
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input [B, C, 1, T], got {tuple(x.shape)}")
        return self.conv(x)


class BaseTCN(nn.Module):
    def __init__(self):
        super(BaseTCN, self).__init__()

    def inference(self, *args, **kwargs):
        return self(*args, inference=True, **kwargs)

    def reset_buffers(self):
        def _reset_buffer(x):
            if isinstance(x, TemporalPad1d):
                x.reset_buffer()
        self.apply(_reset_buffer)

    def get_buffers(self):
        buffers = []

        def _get_buffers(x):
            if isinstance(x, TemporalPad1d):
                buffers.append(x.buffer.clone())
        self.apply(_get_buffers)
        return buffers

    def set_buffers(self, buffers):
        buffers = list(buffers)

        def _set_buffers(x):
            if isinstance(x, TemporalPad1d):
                x.buffer = buffers.pop(0)
        self.apply(_set_buffers)

    def get_in_buffers(self, *args, **kwargs):
        buffers = self.get_buffers()
        buffer_io = BufferIO(in_buffers=None)
        self(*args, inference=True, buffer_io=buffer_io, **kwargs)
        in_buffers = buffer_io.internal_buffers
        self.set_buffers(buffers)
        return in_buffers


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
        use_gate
    ):
        super(TemporalBlock, self).__init__()

        # 第一版量化版：先只支持最稳配置
        if use_norm is not None:
            raise NotImplementedError(
                "Quantized first version only supports use_norm=None."
            )
        if activation != 'relu':
            raise NotImplementedError(
                "Quantized first version only supports activation='relu'."
            )
        if embedding_shapes is not None:
            raise NotImplementedError(
                "Quantized first version only supports embedding_shapes=None."
            )
        if use_gate:
            raise NotImplementedError(
                "Quantized first version only supports use_gate=False."
            )

        self.use_norm = None
        self.activation = activation
        self.kernel_initializer = kerner_initializer
        self.embedding_shapes = None
        self.embedding_mode = embedding_mode
        self.use_gate = False
        self.causal = causal

        self.conv1 = TemporalConv1d(
            in_channels=n_inputs,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            causal=self.causal,
            input_quant=Int8ActPerTensorFloat,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            return_quant_tensor=False,
        )

        self.conv2 = TemporalConv1d(
            in_channels=n_outputs,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            causal=self.causal,
            input_quant=Int8ActPerTensorFloat,
            weight_quant=Int8WeightPerTensorFloat,
            bias_quant=Int32Bias,
            return_quant_tensor=False,
        )

        self.activation1 = make_quant_act()
        self.activation2 = make_quant_act()
        self.activation_final = make_quant_act()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # self.downsample = (
        #     make_quant_pointwise(n_inputs, n_outputs, bias=True)
        #     if n_inputs != n_outputs else None
        # )

        self.downsample = (
            make_quant_pointwise(n_inputs, n_outputs, bias=True)
            if n_inputs != n_outputs else nn.Identity()
        )

        self.init_weights()

    def init_weights(self):
       initialize, kwargs = get_kernel_init_fn(
           name=self.kernel_initializer,
           activation=self.activation,
       )

       initialize(self.conv1.conv.weight, **kwargs)
       initialize(self.conv2.conv.weight, **kwargs)

       if not isinstance(self.downsample, nn.Identity):
           initialize(self.downsample.weight, **kwargs)

    def forward(
        self,
        x,
        embeddings=None,
        inference=False,
        buffer_io=None,
    ):
        if embeddings is not None:
            raise NotImplementedError(
                "Quantized first version does not support embeddings."
            )

        out = self.conv1(
            x,
            inference=inference,
            buffer_io=buffer_io,
        )
        out = self.activation1(out)
        out = self.dropout1(out)

        out = self.conv2(
            out,
            inference=inference,
            buffer_io=buffer_io,
        )
        out = self.activation2(out)
        out = self.dropout2(out)

        # res = x if self.downsample is None else self.downsample(x)
        res = self.downsample(x)
        out = out + res
        out = self.activation_final(out)

        # 保持和原接口兼容
        return out, out


class QTCN(BaseTCN):
    def __init__(
        self,
        num_inputs: int,
        num_channels: ArrayLike,
        kernel_size: int = 4,
        dilations: Optional[ArrayLike] = None,
        dilation_reset: Optional[int] = None,
        dropout: float = 0.1,
        causal: bool = True,
        use_norm: str = None,
        activation: str = 'relu',
        kernel_initializer: str = 'xavier_uniform',
        use_skip_connections: bool = False,
        input_shape: str = 'NCL',
        embedding_shapes: Optional[ArrayLike] = None,
        embedding_mode: str = 'add',
        use_gate: bool = False,
        lookahead=0,
        output_projection: Optional[int] = None,
        output_activation: Optional[str] = None,
    ):
        super(QTCN, self).__init__()

        if lookahead > 0:
            raise ValueError(
                """
                The lookahead parameter is deprecated and must be set to 0.
                """
            )

        if dilations is not None and len(dilations) != len(num_channels):
            raise ValueError("Length of dilations must match length of num_channels")

        self.allowed_norm_values = [None]
        self.allowed_input_shapes = ['NCL', 'NLC']

        _check_generic_input_arg(causal, 'causal', [True, False])
        _check_generic_input_arg(use_norm, 'use_norm', self.allowed_norm_values)
        _check_activation_arg(activation, 'activation')
        _check_generic_input_arg(kernel_initializer, 'kernel_initializer', kernel_init_fn.keys())
        _check_generic_input_arg(use_skip_connections, 'use_skip_connections', [True, False])
        _check_generic_input_arg(input_shape, 'input_shape', self.allowed_input_shapes)
        _check_generic_input_arg(embedding_mode, 'embedding_mode', ['add', 'concat'])
        _check_generic_input_arg(use_gate, 'use_gate', [True, False])
        _check_activation_arg(output_activation, 'output_activation')

        if activation != 'relu':
            raise NotImplementedError(
                "Quantized first version only supports activation='relu'."
            )
        if embedding_shapes is not None:
            raise NotImplementedError(
                "Quantized first version only supports embedding_shapes=None."
            )
        if use_gate:
            raise NotImplementedError(
                "Quantized first version only supports use_gate=False."
            )
        if output_activation is not None:
            raise NotImplementedError(
                "Quantized first version only supports output_activation=None."
            )

        if dilations is None:
            if dilation_reset is None:
                dilations = [2 ** i for i in range(len(num_channels))]
            else:
                dilation_reset = int(np.log2(dilation_reset * 2))
                dilations = [
                    2 ** (i % dilation_reset)
                    for i in range(len(num_channels))
                ]

        self.dilations = dilations
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.use_skip_connections = use_skip_connections
        self.input_shape = input_shape
        self.embedding_shapes = None
        self.embedding_mode = embedding_mode
        self.use_gate = False
        self.causal = causal
        self.output_projection = output_projection
        self.output_activation = None

        self.quant_in = make_quant_identity()

        if use_skip_connections:
            self.downsample_skip_connection = nn.ModuleList()
            for i in range(len(num_channels)):
                if num_channels[i] != num_channels[-1]:
                    self.downsample_skip_connection.append(
                        make_quant_pointwise(num_channels[i], num_channels[-1], bias=True)
                    )
                else:
                    self.downsample_skip_connection.append(None)
            self.init_skip_connection_weights()
            self.activation_skip_out = make_quant_act()
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
                    embedding_mode=self.embedding_mode,
                    use_gate=False,
                )
            )

        self.network = nn.ModuleList(layers)

        if self.output_projection is not None:
            self.projection_out = make_quant_pointwise(
                num_channels[-1],
                self.output_projection,
                bias=True,
            )
            self.init_projection_weights()
        else:
            self.projection_out = None

        self.activation_out = None

        if self.causal:
            self.reset_buffers()

    def init_skip_connection_weights(self):
        initialize, kwargs = get_kernel_init_fn(
            name=self.kernel_initializer,
            activation=self.activation,
        )
        for layer in self.downsample_skip_connection:
            if layer is not None:
                initialize(layer.weight, **kwargs)

    def init_projection_weights(self):
        initialize, kwargs = get_kernel_init_fn(
            name=self.kernel_initializer,
            activation=self.activation,
        )
        initialize(self.projection_out.weight, **kwargs)

    def forward(
        self,
        x,
        embeddings=None,
        inference=False,
        buffer_io=None,
    ):
        if inference and not self.causal:
            raise ValueError(
                """
                Streaming inference is only supported for causal networks.
                """
            )
    
        if embeddings is not None:
            raise NotImplementedError(
                "Quantized first version does not support embeddings."
            )
    
        if self.input_shape == 'NLC':
            x = x.transpose(1, 2)   # [N, L, C] -> [N, C, L]
    
        x = self.quant_in(x)        # still [N, C, L]
        x = x.unsqueeze(2)          # only once: [N, C, 1, L]
    
        if self.use_skip_connections:
            skip_connections = []
    
            for index, layer in enumerate(self.network):
                x, skip_out = layer(
                    x,
                    embeddings=None,
                    inference=inference,
                    buffer_io=buffer_io,
                )
                if self.downsample_skip_connection[index] is not None:
                    skip_out = self.downsample_skip_connection[index](skip_out)
                if index < len(self.network) - 1:
                    skip_connections.append(skip_out)
    
            skip_connections.append(x)
            x = torch.stack(skip_connections, dim=0).sum(dim=0)
            x = self.activation_skip_out(x)
    
        else:
            for layer in self.network:
                x, _ = layer(
                    x,
                    embeddings=None,
                    inference=inference,
                    buffer_io=buffer_io,
                )
    
        if self.projection_out is not None:
            x = self.projection_out(x)
    
        x = x.squeeze(2)            # only once at the end: [N, C, L]
    
        if self.input_shape == 'NLC':
            x = x.transpose(1, 2)
    
        return x