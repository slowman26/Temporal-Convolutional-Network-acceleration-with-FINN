import torch
import torch.nn as nn


class TemporalPad1dExport(nn.Module):
    """
    Export-only causal left zero padding.

    输入:
        x: [B, C, T]

    输出:
        y: [B, C, T + pad_len]

    说明:
        - 只做左侧 zero padding
        - forward 中不包含任何 if 分支
        - 用于静态导出，不用于 streaming buffer
    """

    def __init__(
        self,
        padding: int,
        in_channels: int,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.pad_len = int(padding)
        self.in_channels = int(in_channels)

        self.register_buffer(
            "left_zeros",
            torch.zeros(1, in_channels, padding, device=device, dtype=dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.left_zeros.expand(x.shape[0], -1, -1)
        y = torch.cat([z, x], dim=-1)
        return y