"""Runtime, memory, and model-size benchmark helpers."""

from __future__ import annotations

from typing import Any

import torch


BYTES_PER_MIB = 1024.0 * 1024.0


def tensor_size_bytes(tensor: torch.Tensor) -> int:
    """Return the storage size of a tensor-like value in bytes."""
    return int(tensor.numel() * tensor.element_size())


def model_size_summary(model) -> dict[str, Any]:
    """Summarize parameter, buffer, and state-dict storage sizes.

    `param_size_mb` measures learnable parameters only. `buffer_size_mb` includes
    non-parameter tensors such as entropy-model CDF tables. `state_dict_size_mb`
    is an in-memory tensor-size estimate for all tensors in the state dict; it is
    usually a better deployment proxy for learned image compression models than
    parameter size alone.
    """
    param_bytes = sum(tensor_size_bytes(param) for param in model.parameters())
    buffer_bytes = sum(tensor_size_bytes(buffer) for buffer in model.buffers())

    state_dict_bytes = 0
    state_dict_tensors = 0
    for value in model.state_dict().values():
        if torch.is_tensor(value):
            state_dict_bytes += tensor_size_bytes(value)
            state_dict_tensors += 1

    num_parameters = sum(int(param.numel()) for param in model.parameters())
    num_buffers = sum(int(buffer.numel()) for buffer in model.buffers())

    return {
        "num_parameters": num_parameters,
        "num_buffer_values": num_buffers,
        "num_state_dict_tensors": state_dict_tensors,
        "param_size_mb": param_bytes / BYTES_PER_MIB,
        "buffer_size_mb": buffer_bytes / BYTES_PER_MIB,
        "state_dict_size_mb": state_dict_bytes / BYTES_PER_MIB,
    }


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device) / BYTES_PER_MIB)


def compression_ratio_from_bpp(bpp: float, source_bits_per_pixel: int = 24) -> float:
    """Compute raw RGB compression ratio from compressed bpp."""
    if bpp <= 0:
        return float("inf")
    return float(source_bits_per_pixel) / float(bpp)
