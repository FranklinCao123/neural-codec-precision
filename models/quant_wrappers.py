"""Model wrappers used by precision experiments."""

from __future__ import annotations

import torch
from torch import nn


class AutoCastInput(nn.Module):
    """Cast tensor inputs to the wrapped module's first floating-point dtype."""

    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module

    def forward(self, x, *args, **kwargs):
        dtype = module_dtype(self.module)
        if torch.is_tensor(x) and dtype is not None and x.is_floating_point():
            x = x.to(dtype=dtype)
        return self.module(x, *args, **kwargs)


def module_dtype(module: nn.Module) -> torch.dtype | None:
    for param in module.parameters(recurse=True):
        return param.dtype
    for buffer in module.buffers(recurse=True):
        if torch.is_floating_point(buffer):
            return buffer.dtype
    return None
