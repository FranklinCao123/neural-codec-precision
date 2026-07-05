"""Post-training INT8 quantization helpers.

This module implements INT8 weight fake-quantization for learned image
compression experiments. We quantize selected module weights to INT8 and
immediately dequantize them back to FP32 so CompressAI's entropy-coding path can
run unchanged while the model still carries INT8 weight quantization error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class QuantizedTensorStats:
    name: str
    numel: int
    original_bytes: int
    simulated_quantized_bytes: int


def prepare_ptq(model, config: dict):
    """Prepare a model for PTQ.

    Fake weight quantization does not need calibration because it only quantizes
    static weights. Activation PTQ can be added later as a separate experiment.
    """
    return model


def convert_ptq(model, config: dict | None = None):
    """Apply INT8 fake-quantization according to a config."""
    config = config or {}
    precision_cfg = config.get("precision", {})
    modules = precision_cfg.get("int8_modules") or precision_cfg.get("modules")
    if not modules:
        modules = ["g_a", "g_s", "h_a", "h_s"]

    granularity = precision_cfg.get("weight_quantization", "per_channel")
    stats = fake_quantize_modules(model, modules, granularity=granularity)
    model._quantization_summary = _summarize_quantization(stats, modules, granularity)
    return model


def fake_quantize_modules(
    model,
    modules: list[str] | tuple[str, ...],
    granularity: str = "per_channel",
):
    """Quantize selected module parameters to INT8 then dequantize to FP32."""
    stats: list[QuantizedTensorStats] = []
    with torch.no_grad():
        for module_name in modules:
            if not hasattr(model, module_name):
                raise ValueError(f"Model does not have module {module_name!r}")
            module = getattr(model, module_name)
            for param_name, param in module.named_parameters(recurse=True):
                if not param.is_floating_point():
                    continue
                full_name = f"{module_name}.{param_name}"
                dequantized, quantized_bytes = fake_quantize_tensor(
                    param.detach(),
                    granularity=granularity,
                )
                param.copy_(dequantized.to(dtype=param.dtype))
                stats.append(
                    QuantizedTensorStats(
                        name=full_name,
                        numel=int(param.numel()),
                        original_bytes=int(param.numel() * param.element_size()),
                        simulated_quantized_bytes=quantized_bytes,
                    )
                )
    return stats


def fake_quantize_tensor(tensor: torch.Tensor, granularity: str = "per_channel"):
    """Return a dequantized tensor and simulated INT8 storage bytes."""
    if tensor.numel() == 0:
        return tensor.clone(), 0
    if granularity == "per_channel" and tensor.ndim >= 2:
        return _fake_quantize_per_channel(tensor)
    if granularity == "per_tensor":
        return _fake_quantize_per_tensor(tensor)
    if granularity == "per_channel":
        return _fake_quantize_per_tensor(tensor)
    raise ValueError(f"Unsupported weight quantization granularity: {granularity!r}")


def _fake_quantize_per_tensor(tensor: torch.Tensor):
    max_abs = tensor.detach().abs().max()
    scale = torch.clamp(max_abs / 127.0, min=torch.finfo(tensor.dtype).eps)
    q = torch.clamp(torch.round(tensor / scale), -127, 127)
    dequantized = q * scale
    quantized_bytes = int(tensor.numel()) + 4
    return dequantized, quantized_bytes


def _fake_quantize_per_channel(tensor: torch.Tensor):
    view_shape = [tensor.shape[0]] + [1] * (tensor.ndim - 1)
    reduce_dims = tuple(range(1, tensor.ndim))
    max_abs = tensor.detach().abs().amax(dim=reduce_dims, keepdim=True)
    scale = torch.clamp(max_abs / 127.0, min=torch.finfo(tensor.dtype).eps)
    q = torch.clamp(torch.round(tensor / scale), -127, 127)
    dequantized = q * scale
    num_channels = int(tensor.reshape(tensor.shape[0], -1).shape[0])
    quantized_bytes = int(tensor.numel()) + num_channels * 4
    return dequantized.reshape_as(tensor), quantized_bytes


def _summarize_quantization(
    stats: list[QuantizedTensorStats],
    modules: list[str] | tuple[str, ...],
    granularity: str,
) -> dict[str, Any]:
    original_bytes = sum(item.original_bytes for item in stats)
    quantized_bytes = sum(item.simulated_quantized_bytes for item in stats)
    return {
        "quantization_method": "int8_weight_fake_quant",
        "quantized_modules": list(modules),
        "weight_quantization": granularity,
        "num_quantized_tensors": len(stats),
        "num_quantized_values": sum(item.numel for item in stats),
        "quantized_weight_original_mb": original_bytes / (1024.0 * 1024.0),
        "quantized_weight_simulated_int8_mb": quantized_bytes / (1024.0 * 1024.0),
        "quantized_weight_size_reduction": (
            1.0 - float(quantized_bytes) / float(original_bytes)
            if original_bytes > 0
            else 0.0
        ),
    }
