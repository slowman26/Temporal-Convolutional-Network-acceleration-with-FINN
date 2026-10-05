import torch
import torch.nn as nn


# class TemporalPad2dExport(nn.Module):
#     def __init__(
#         self,
#         padding: int,
#         in_channels: int,
#         device=None,
#         dtype=None,
#     ):
#         super().__init__()

#         self.pad_len = int(padding)
#         self.in_channels = int(in_channels)

#         self.register_buffer(
#             "left_zeros",
#             torch.zeros(1, in_channels, 1, self.pad_len, device=device, dtype=dtype),
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         z = self.left_zeros.expand(x.shape[0], -1, -1, -1)
#         y = torch.cat([z, x], dim=-1)
#         return y
    
class TemporalPad2dExport(nn.Module):
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
            torch.zeros(1, in_channels, 1, self.pad_len, device=device, dtype=dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.left_zeros.expand(x.shape[0], -1, -1, -1)
        return torch.cat([z, x], dim=-1)