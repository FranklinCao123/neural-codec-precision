"""Model wrappers used by precision experiments."""

from __future__ import annotations

import torch
from torch import nn


class AutoCastInput(nn.Module):
    """Cast tensor inputs to module dtype and optionally cast outputs."""

    def __init__(self, module: nn.Module, output_dtype: torch.dtype | None = None):
        super().__init__()
        self.module = module
        self.output_dtype = output_dtype

    def forward(self, x, *args, **kwargs):
        dtype = module_dtype(self.module)
        if torch.is_tensor(x) and dtype is not None and x.is_floating_point():
            x = x.to(dtype=dtype)
        output = self.module(x, *args, **kwargs)
        if self.output_dtype is None:
            return output
        return cast_floating_tensors(output, self.output_dtype)


def cast_floating_tensors(value, dtype: torch.dtype):
    if torch.is_tensor(value):
        if value.is_floating_point():
            return value.to(dtype=dtype)
        return value
    if isinstance(value, tuple):
        return tuple(cast_floating_tensors(item, dtype) for item in value)
    if isinstance(value, list):
        return [cast_floating_tensors(item, dtype) for item in value]
    if isinstance(value, dict):
        return {key: cast_floating_tensors(item, dtype) for key, item in value.items()}
    return value


def module_dtype(module: nn.Module) -> torch.dtype | None:
    for param in module.parameters(recurse=True):
        return param.dtype
    for buffer in module.buffers(recurse=True):
        if torch.is_floating_point(buffer):
            return buffer.dtype
    return None
