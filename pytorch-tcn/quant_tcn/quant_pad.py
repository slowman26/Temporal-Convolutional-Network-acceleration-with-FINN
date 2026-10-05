import torch
import torch.nn as nn

from .quant_buffer import BufferIO
from typing import Optional, Union

PADDING_MODES = [
    "zeros",
    "reflect",
    "replicate",
    "circular",
]


class TemporalPad1d(nn.Module):
    def __init__(
        self,
        padding: int,
        in_channels: int,
        buffer: Optional[Union[float, torch.Tensor]] = None,
        padding_mode: str = "zeros",
        causal: bool = False,
    ):
        super().__init__()

        if not isinstance(padding, int):
            raise ValueError(
                f"padding must be an integer, but got {type(padding)}."
            )
        if in_channels is None:
            raise ValueError("in_channels must be specified.")
        if padding_mode not in PADDING_MODES:
            raise ValueError(
                f"padding_mode must be one of {PADDING_MODES}, but got {padding_mode}."
            )

        self.pad_len = padding
        self.in_channels = in_channels
        self.causal = causal
        self.padding_mode = padding_mode

        if causal:
            self.left_padding = self.pad_len
            self.right_padding = 0
        else:
            self.left_padding = self.pad_len // 2
            self.right_padding = self.pad_len - self.left_padding

        # 普通 3D eager 路径
        if padding_mode == "zeros":
            self.pad = nn.ConstantPad1d(
                (self.left_padding, self.right_padding), 0.0
            )
        elif padding_mode == "reflect":
            self.pad = nn.ReflectionPad1d(
                (self.left_padding, self.right_padding)
            )
        elif padding_mode == "replicate":
            self.pad = nn.ReplicationPad1d(
                (self.left_padding, self.right_padding)
            )
        elif padding_mode == "circular":
            self.pad = nn.CircularPad1d(
                (self.left_padding, self.right_padding)
            )

        # 3D 静态零张量
        self.register_buffer(
            "left_zeros",
            torch.zeros(1, in_channels, self.left_padding),
        )
        self.register_buffer(
            "right_zeros",
            torch.zeros(1, in_channels, self.right_padding),
        )

        # 4D 静态零张量
        self.register_buffer(
            "left_zeros_4d",
            torch.zeros(1, in_channels, 1, self.left_padding),
        )
        self.register_buffer(
            "right_zeros_4d",
            torch.zeros(1, in_channels, 1, self.right_padding),
        )

        # streaming inference buffer 仍按 3D 维护
        if buffer is None:
            buffer = torch.zeros(1, in_channels, self.pad_len)
        elif isinstance(buffer, (int, float)):
            buffer = torch.full(
                size=(1, in_channels, self.pad_len),
                fill_value=buffer,
            )
        elif not isinstance(buffer, torch.Tensor):
            raise ValueError(
                f"The argument 'buffer' must be None, float, int, or torch.Tensor, but got {type(buffer)}."
            )

        self.register_buffer("buffer", buffer)

    def _concat_nonempty(self, tensors, dim: int):
        parts = [t for t in tensors if t is not None]
        if len(parts) == 0:
            raise ValueError("No tensors to concatenate.")
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=dim)

    def _get_left_zeros_3d(self, x: torch.Tensor):
        if self.left_padding == 0:
            return None
        return self.left_zeros.to(device=x.device, dtype=x.dtype).expand(
            x.shape[0], -1, -1
        )

    def _get_right_zeros_3d(self, x: torch.Tensor):
        if self.right_padding == 0:
            return None
        return self.right_zeros.to(device=x.device, dtype=x.dtype).expand(
            x.shape[0], -1, -1
        )

    def _get_left_zeros_4d(self, x: torch.Tensor):
        if self.left_padding == 0:
            return None
        return self.left_zeros_4d.to(device=x.device, dtype=x.dtype).expand(
            x.shape[0], -1, -1, -1
        )

    def _get_right_zeros_4d(self, x: torch.Tensor):
        if self.right_padding == 0:
            return None
        return self.right_zeros_4d.to(device=x.device, dtype=x.dtype).expand(
            x.shape[0], -1, -1, -1
        )

    def pad_export_onnx(self, x: torch.Tensor):
        """3D ONNX/FINN 导出路径: x = [N, C, L]"""
        if self.padding_mode != "zeros":
            raise NotImplementedError(
                "For ONNX/FINN export, please use padding_mode='zeros'."
            )
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [N, C, L], got {tuple(x.shape)}")

        left = self._get_left_zeros_3d(x)
        right = self._get_right_zeros_3d(x)
        return self._concat_nonempty([left, x, right], dim=-1)

    def pad_export_onnx_4d(self, x: torch.Tensor):
        """4D ONNX/FINN 导出路径: x = [N, C, 1, L]"""
        if self.padding_mode != "zeros":
            raise NotImplementedError(
                "For ONNX/FINN export, please use padding_mode='zeros'."
            )
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input [N, C, 1, L], got {tuple(x.shape)}")

        left = self._get_left_zeros_4d(x)
        right = self._get_right_zeros_4d(x)
        return self._concat_nonempty([left, x, right], dim=-1)

    def pad_4d(self, x: torch.Tensor):
        """普通 PyTorch 4D 路径: x = [N, C, 1, L]"""
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input [N, C, 1, L], got {tuple(x.shape)}")
        if self.padding_mode != "zeros":
            raise NotImplementedError(
                "Only zeros padding is supported in the 4D path."
            )

        left = self._get_left_zeros_4d(x)
        right = self._get_right_zeros_4d(x)
        return self._concat_nonempty([left, x, right], dim=-1)

    def pad_inference(
        self,
        x: torch.Tensor,
        buffer_io: Optional[BufferIO] = None,
    ):
        """streaming inference 仍只支持 3D: x = [1, C, L]"""
        if not self.causal:
            raise ValueError(
                "Streaming inference is only supported for causal convolutions."
            )
        if x.dim() != 3:
            raise ValueError(
                f"Streaming inference expects 3D input [1, C, L], got {tuple(x.shape)}"
            )
        if x.shape[0] != 1:
            raise ValueError(
                f"Streaming inference requires batch size 1, but got {x.shape[0]}."
            )

        if buffer_io is None:
            in_buffer = self.buffer
        else:
            in_buffer = buffer_io.next_in_buffer()
            if in_buffer is None:
                in_buffer = self.buffer
                buffer_io.append_internal_buffer(in_buffer)

        x = torch.cat((in_buffer, x), dim=-1)

        out_buffer = x[..., -self.pad_len :]
        if buffer_io is None:
            self.buffer = out_buffer
        else:
            buffer_io.append_out_buffer(out_buffer)

        return x

    def forward(
        self,
        x: torch.Tensor,
        inference: bool = False,
        buffer_io: Optional[BufferIO] = None,
    ):
        if inference:
            return self.pad_inference(x, buffer_io=buffer_io)

        # 导出 ONNX 时走静态拼接路径，避免 ONNX Pad
        if torch.onnx.is_in_onnx_export():
            if x.dim() == 3:
                return self.pad_export_onnx(x)
            elif x.dim() == 4:
                return self.pad_export_onnx_4d(x)
            else:
                raise ValueError(
                    f"ONNX export expects 3D or 4D input, got {tuple(x.shape)}"
                )

        # 普通 PyTorch 路径
        if x.dim() == 3:
            return self.pad(x)
        elif x.dim() == 4:
            return self.pad_4d(x)
        else:
            raise ValueError(
                f"TemporalPad1d expects 3D [N, C, L] or 4D [N, C, 1, L], got {tuple(x.shape)}"
            )

    def reset_buffer(self):
        self.buffer.zero_()
        if self.buffer.shape[-1] != self.pad_len:
            raise ValueError(
                f"Buffer shape {self.buffer.shape} does not match expected last dim {self.pad_len}."
            )