import torch
import torch.nn as nn


class TemporalPad2dTrain(nn.Module):
    """
    训练 / 普通推理版
    支持任意 batch
    """
    def __init__(self, padding: int, in_channels: int, device=None, dtype=None):
        super().__init__()
        self.pad_len = int(padding)
        self.in_channels = int(in_channels)

        self.register_buffer(
            "left_zeros",
            torch.zeros(1, in_channels, 1, self.pad_len, device=device, dtype=dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.left_zeros.expand(x.shape[0], -1, -1, -1)
        return torch.cat([z, x], dim=-1)


class TemporalPad2dExport(nn.Module):
    """
    导出版
    固定 batch=1，尽量减少动态 shape 子图
    """
    def __init__(self, padding: int, in_channels: int, device=None, dtype=None):
        super().__init__()
        self.pad_len = int(padding)
        self.in_channels = int(in_channels)

        self.register_buffer(
            "left_zeros",
            torch.zeros(1, in_channels, 1, self.pad_len, device=device, dtype=dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.left_zeros, x], dim=-1)