"""Activation fake-quantization helpers for precision sensitivity tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import nn

DEFAULT_ACTIVATION_MODULES = ("g_a", "g_s", "h_a", "h_s")


@dataclass
class FakeQuantStats:
    module: str
    dtype: str


class FakeQuantOutput(nn.Module):
    """Apply fake quantization to floating outputs of a wrapped module."""

    def __init__(self, module: nn.Module, quantize_fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.module = module
        self.quantize_fn = quantize_fn

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        return _map_floating_tensors(output, self.quantize_fn)


def apply_activation_fake_quant(model, config: dict):
    """Wrap selected top-level modules with activation fake quantization."""
    precision_cfg = config.get("precision", {})
    modules = precision_cfg.get("activation_modules") or precision_cfg.get("modules")
    if not modules:
        modules = list(DEFAULT_ACTIVATION_MODULES)

    fake_dtype = str(precision_cfg.get("fake_dtype", precision_cfg.get("mode", ""))).lower()
    if fake_dtype in {"bf16", "bfloat16"}:
        quantize_fn = fake_bf16
        method = "bf16_activation_fake_quant"
    elif _is_fixed_point_dtype(fake_dtype):
        num_bits = _fixed_point_bits(fake_dtype, precision_cfg)
        fractional_bits = precision_cfg.get("fractional_bits")
        quantize_fn = lambda tensor: fake_fixed_point(
            tensor,
            num_bits=num_bits,
            fractional_bits=fractional_bits,
        )
        method = f"int{num_bits}_activation_fixed_point_fake_quant"
    elif fake_dtype in {"fp8", "fp8_e4m3", "e4m3"}:
        quantize_fn = lambda tensor: fake_fp8(tensor, exponent_bits=4, mantissa_bits=3)
        method = "fp8_e4m3_activation_fake_quant"
    elif fake_dtype in {"fp8_e5m2", "e5m2"}:
        quantize_fn = lambda tensor: fake_fp8(tensor, exponent_bits=5, mantissa_bits=2)
        method = "fp8_e5m2_activation_fake_quant"
    else:
        raise ValueError(f"Unsupported activation fake dtype: {fake_dtype!r}")

    stats: list[FakeQuantStats] = []
    for module_name in modules:
        if not hasattr(model, module_name):
            raise ValueError(f"Model does not have module {module_name!r}")
        module = getattr(model, module_name)
        if isinstance(module, FakeQuantOutput):
            module = module.module
        setattr(model, module_name, FakeQuantOutput(module, quantize_fn))
        stats.append(FakeQuantStats(module=module_name, dtype=fake_dtype))

    model._quantization_summary = {
        "quantization_method": method,
        "activation_quantized_modules": [item.module for item in stats],
        "activation_fake_dtype": fake_dtype,
        "num_activation_fake_quant_modules": len(stats),
    }
    if _is_fixed_point_dtype(fake_dtype):
        model._quantization_summary["fixed_point_bits"] = _fixed_point_bits(
            fake_dtype,
            precision_cfg,
        )
        if precision_cfg.get("fractional_bits") is not None:
            model._quantization_summary["fixed_point_fractional_bits"] = int(
                precision_cfg["fractional_bits"]
            )
    return model


def fake_bf16(tensor: torch.Tensor) -> torch.Tensor:
    """Round a floating tensor through BF16 and return FP32/normal dtype values."""
    return tensor.to(torch.bfloat16).to(torch.float32).to(tensor.dtype)


def _is_fixed_point_dtype(fake_dtype: str) -> bool:
    return fake_dtype in {
        "int8",
        "fixed_int8",
        "fixed8",
        "int12",
        "fixed_int12",
        "fixed12",
        "int16",
        "fixed_int16",
        "fixed16",
    }


def _fixed_point_bits(fake_dtype: str, precision_cfg: dict) -> int:
    if precision_cfg.get("fixed_point_bits") is not None:
        return int(precision_cfg["fixed_point_bits"])
    for bits in (8, 12, 16):
        if str(bits) in fake_dtype:
            return bits
    raise ValueError(f"Cannot infer fixed-point bit width from fake dtype: {fake_dtype!r}")


def fake_fixed_point(
    tensor: torch.Tensor,
    num_bits: int,
    fractional_bits: int | None = None,
) -> torch.Tensor:
    """Symmetric signed fixed-point fake quantization."""
    if tensor.numel() == 0:
        return tensor
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -(1 << (num_bits - 1))
    if fractional_bits is None:
        max_abs = tensor.detach().abs().max()
        scale = torch.clamp(max_abs / float(qmax), min=torch.finfo(tensor.dtype).eps)
    else:
        scale = torch.tensor(
            2.0 ** (-int(fractional_bits)),
            dtype=tensor.dtype,
            device=tensor.device,
        )
    quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    return quantized * scale


def fake_fp8(tensor: torch.Tensor, exponent_bits: int = 4, mantissa_bits: int = 3) -> torch.Tensor:
    """Approximate FP8 fake quantization with IEEE-like exponent/mantissa rounding."""
    if tensor.numel() == 0:
        return tensor
    if exponent_bits < 2 or mantissa_bits < 1:
        raise ValueError("FP8 fake quantization needs exponent_bits >= 2 and mantissa_bits >= 1.")

    x = tensor.float()
    sign = torch.sign(x)
    abs_x = x.abs()
    zero = abs_x == 0

    bias = (1 << (exponent_bits - 1)) - 1
    min_exp = 1 - bias
    max_exp = ((1 << exponent_bits) - 2) - bias

    safe_abs = torch.where(zero, torch.ones_like(abs_x), abs_x)
    exponent = torch.floor(torch.log2(safe_abs)).clamp(min_exp, max_exp)
    step = torch.pow(2.0, exponent - mantissa_bits)
    max_value = (2.0 - 2.0 ** (-mantissa_bits)) * (2.0 ** max_exp)

    rounded = torch.round(safe_abs / step) * step
    rounded = torch.clamp(rounded, 0.0, max_value)
    rounded = torch.where(zero, torch.zeros_like(rounded), rounded)
    return (sign * rounded).to(dtype=tensor.dtype)


def _map_floating_tensors(value: Any, fn: Callable[[torch.Tensor], torch.Tensor]):
    if torch.is_tensor(value):
        if value.is_floating_point():
            return fn(value)
        return value
    if isinstance(value, tuple):
        return tuple(_map_floating_tensors(item, fn) for item in value)
    if isinstance(value, list):
        return [_map_floating_tensors(item, fn) for item in value]
    if isinstance(value, dict):
        return {key: _map_floating_tensors(item, fn) for key, item in value.items()}
    return value
