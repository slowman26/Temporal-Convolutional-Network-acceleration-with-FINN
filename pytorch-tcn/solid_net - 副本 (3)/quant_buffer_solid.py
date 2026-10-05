import torch
import torch.nn as nn
from typing import Optional, Tuple


class FixedCausalBuffer2d(nn.Module):
    """
    Fixed-size buffer for causal streaming temporal convolution (4D version).

    用途：
        给某一层 causal temporal conv 提供固定长度的历史状态。

    输入：
        x:      [B, C, 1, T]
        state:  [B, C, 1, L] 或 None

    输出：
        x_cat:      [B, C, 1, L + T]   = concat(state, x)
        new_state:  [B, C, 1, L]

    其中：
        L = dilation * (kernel_size - 1)

    说明：
        1. 内部统一使用 4D: [B, C, 1, T]
        2. 对 FINN / ONNX，更推荐显式传 state
        3. update_internal=True 时，会把 new_state 写回模块内部 self.state
        4. 内部状态模式更适合本地 PyTorch 调试；部署时建议用显式 state_in/state_out
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int = 1,
        batch_size: int = 1,
        device=None,
        dtype=None,
        persistent: bool = False,
    ):
        super().__init__()

        if channels <= 0:
            raise ValueError(f"channels must be > 0, but got {channels}")
        if kernel_size <= 0:
            raise ValueError(f"kernel_size must be > 0, but got {kernel_size}")
        if dilation <= 0:
            raise ValueError(f"dilation must be > 0, but got {dilation}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, but got {batch_size}")

        self.channels = channels
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.batch_size = batch_size
        self.buffer_len = dilation * (kernel_size - 1)

        factory_kwargs = {"device": device, "dtype": dtype}

        self.register_buffer(
            "state",
            torch.zeros(batch_size, channels, 1, self.buffer_len, **factory_kwargs),
            persistent=persistent,
        )

    def extra_repr(self) -> str:
        return (
            f"channels={self.channels}, "
            f"kernel_size={self.kernel_size}, "
            f"dilation={self.dilation}, "
            f"buffer_len={self.buffer_len}, "
            f"batch_size={self.batch_size}"
        )

    def _check_x(self, x: torch.Tensor) -> None:
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, but got {type(x)}")
        if x.ndim != 4:
            raise ValueError(f"x must have shape [B, C, 1, T], but got {tuple(x.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError(
                f"x.shape[1] must be {self.channels}, but got {x.shape[1]}"
            )
        if x.shape[2] != 1:
            raise ValueError(
                f"x.shape[2] must be 1, but got {x.shape[2]}"
            )

    def _check_state(self, state: torch.Tensor, batch_size: int) -> None:
        if not isinstance(state, torch.Tensor):
            raise TypeError(f"state must be a torch.Tensor, but got {type(state)}")
        expected_shape = (batch_size, self.channels, 1, self.buffer_len)
        if tuple(state.shape) != expected_shape:
            raise ValueError(
                f"state must have shape {expected_shape}, but got {tuple(state.shape)}"
            )

    def make_zeros_state(
        self,
        batch_size: int,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, but got {batch_size}")
        if device is None:
            device = self.state.device
        if dtype is None:
            dtype = self.state.dtype
        return torch.zeros(
            batch_size,
            self.channels,
            1,
            self.buffer_len,
            device=device,
            dtype=dtype,
        )

    def reset_state(
        self,
        batch_size: Optional[int] = None,
        device=None,
        dtype=None,
    ) -> None:
        if batch_size is None:
            batch_size = self.batch_size

        new_state = self.make_zeros_state(
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )

        if tuple(new_state.shape) != tuple(self.state.shape):
            self.state = new_state
            self.batch_size = batch_size
        else:
            with torch.no_grad():
                self.state.zero_()

    def get_state(self) -> torch.Tensor:
        return self.state

    def set_state(self, state: torch.Tensor) -> None:
        self._check_state(state, batch_size=state.shape[0])
        if tuple(state.shape) != tuple(self.state.shape):
            self.state = state.clone()
            self.batch_size = state.shape[0]
        else:
            with torch.no_grad():
                self.state.copy_(state)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        update_internal: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        参数：
            x: [B, C, 1, T]
            state: [B, C, 1, L]；若为 None，则使用模块内部 self.state
            update_internal: 是否把 new_state 写回 self.state

        返回：
            x_cat: [B, C, 1, L+T]
            new_state: [B, C, 1, L]
        """
        self._check_x(x)
        B, C, H, T = x.shape

        if state is None:
            if self.state.shape[0] != B:
                raise ValueError(
                    f"Internal state batch size is {self.state.shape[0]}, "
                    f"but input batch size is {B}. "
                    f"Please call reset_state(batch_size={B}) first, "
                    f"or pass explicit state."
                )
            state = self.state
        else:
            self._check_state(state, batch_size=B)

        if self.buffer_len == 0:
            x_cat = x
            new_state = state[..., :0]
        else:
            x_cat = torch.cat([state, x], dim=-1)
            new_state = x_cat[..., -self.buffer_len:]

        if update_internal:
            if tuple(self.state.shape) != tuple(new_state.shape):
                self.state = new_state.detach().clone()
                self.batch_size = new_state.shape[0]
            else:
                with torch.no_grad():
                    self.state.copy_(new_state.detach())

        return x_cat, new_state