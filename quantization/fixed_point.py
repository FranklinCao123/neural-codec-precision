"""Fixed-point probing helpers.

The probe path materializes integer activation tensors, records their ranges and
storage cost, then dequantizes back to floating point so CompressAI's codec path
can continue unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn


DEFAULT_PROBE_MODULES = ("g_a", "g_s", "h_a", "h_s")


@dataclass
class ModuleProbeStats:
    module: str
    calls: int = 0
    num_values: int = 0
    original_float_bytes: int = 0
    integer_storage_bytes: int = 0
    packed_storage_bits: int = 0
    saturated_values: int = 0
    min_float: float | None = None
    max_float: float | None = None
    min_int: int | None = None
    max_int: int | None = None
    min_scale: float | None = None
    max_scale: float | None = None
    sum_scale: float = 0.0

    def update(
        self,
        tensor: torch.Tensor,
        integer_tensor: torch.Tensor,
        scale: torch.Tensor,
        qmin: int,
        qmax: int,
        storage_dtype: torch.dtype,
    ) -> None:
        self.calls += 1
        numel = int(tensor.numel())
        self.num_values += numel
        self.original_float_bytes += numel * int(tensor.element_size())
        self.integer_storage_bytes += numel * _dtype_size(storage_dtype)
        self.packed_storage_bits += numel * (qmax.bit_length() + 1)

        if numel == 0:
            return
        tensor_detached = tensor.detach()
        integer_detached = integer_tensor.detach()
        scale_value = float(scale.detach().float().item())

        self.saturated_values += int(
            ((integer_detached == qmin) | (integer_detached == qmax)).sum().item()
        )
        self.min_float = _min_optional(self.min_float, float(tensor_detached.min().item()))
        self.max_float = _max_optional(self.max_float, float(tensor_detached.max().item()))
        self.min_int = _min_optional(self.min_int, int(integer_detached.min().item()))
        self.max_int = _max_optional(self.max_int, int(integer_detached.max().item()))
        self.min_scale = _min_optional(self.min_scale, scale_value)
        self.max_scale = _max_optional(self.max_scale, scale_value)
        self.sum_scale += scale_value

    def as_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "calls": self.calls,
            "num_values": self.num_values,
            "original_float_bytes": self.original_float_bytes,
            "integer_storage_bytes": self.integer_storage_bytes,
            "packed_storage_bits": self.packed_storage_bits,
            "packed_storage_bytes": self.packed_storage_bits / 8.0,
            "storage_reduction": (
                1.0 - float(self.integer_storage_bytes) / float(self.original_float_bytes)
                if self.original_float_bytes
                else 0.0
            ),
            "packed_storage_reduction": (
                1.0 - (float(self.packed_storage_bits) / 8.0) / float(self.original_float_bytes)
                if self.original_float_bytes
                else 0.0
            ),
            "saturated_values": self.saturated_values,
            "saturation_ratio": (
                float(self.saturated_values) / float(self.num_values) if self.num_values else 0.0
            ),
            "min_float": self.min_float,
            "max_float": self.max_float,
            "min_int": self.min_int,
            "max_int": self.max_int,
            "min_scale": self.min_scale,
            "max_scale": self.max_scale,
            "avg_scale": self.sum_scale / float(self.calls) if self.calls else None,
        }


@dataclass
class FixedPointProbeCollector:
    num_bits: int
    fractional_bits: int | None = None
    storage_dtype: torch.dtype = torch.int16
    modules: dict[str, ModuleProbeStats] = field(default_factory=dict)

    def stats_for(self, module_name: str) -> ModuleProbeStats:
        if module_name not in self.modules:
            self.modules[module_name] = ModuleProbeStats(module=module_name)
        return self.modules[module_name]

    def as_dict(self) -> dict[str, Any]:
        module_rows = [stats.as_dict() for stats in self.modules.values()]
        total_values = sum(row["num_values"] for row in module_rows)
        total_float_bytes = sum(row["original_float_bytes"] for row in module_rows)
        total_integer_bytes = sum(row["integer_storage_bytes"] for row in module_rows)
        total_packed_bits = sum(row["packed_storage_bits"] for row in module_rows)
        saturated = sum(row["saturated_values"] for row in module_rows)
        return {
            "probe_method": "fixed_point_activation_probe",
            "fixed_point_bits": self.num_bits,
            "fixed_point_fractional_bits": self.fractional_bits,
            "integer_storage_dtype": str(self.storage_dtype).replace("torch.", ""),
            "total_activation_values": total_values,
            "total_original_float_bytes": total_float_bytes,
            "total_integer_storage_bytes": total_integer_bytes,
            "total_packed_storage_bits": total_packed_bits,
            "total_packed_storage_bytes": total_packed_bits / 8.0,
            "total_storage_reduction": (
                1.0 - float(total_integer_bytes) / float(total_float_bytes)
                if total_float_bytes
                else 0.0
            ),
            "total_packed_storage_reduction": (
                1.0 - (float(total_packed_bits) / 8.0) / float(total_float_bytes)
                if total_float_bytes
                else 0.0
            ),
            "total_saturated_values": saturated,
            "total_saturation_ratio": float(saturated) / float(total_values) if total_values else 0.0,
            "modules": module_rows,
        }


class FixedPointProbeOutput(nn.Module):
    """Wrap a module and probe its floating output as fixed-point integers."""

    def __init__(
        self,
        module_name: str,
        module: nn.Module,
        collector: FixedPointProbeCollector,
    ):
        super().__init__()
        self.module_name = module_name
        self.module = module
        self.collector = collector

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        return _map_floating_tensors(output, self._probe_tensor)

    def _probe_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        dequantized, integer_tensor, scale, qmin, qmax = quantize_to_fixed_point_integer(
            tensor,
            num_bits=self.collector.num_bits,
            fractional_bits=self.collector.fractional_bits,
            storage_dtype=self.collector.storage_dtype,
        )
        self.collector.stats_for(self.module_name).update(
            tensor=tensor,
            integer_tensor=integer_tensor,
            scale=scale,
            qmin=qmin,
            qmax=qmax,
            storage_dtype=self.collector.storage_dtype,
        )
        return dequantized


def apply_fixed_point_probe(model, config: dict):
    """Wrap selected modules with fixed-point activation probing."""
    precision_cfg = config.get("precision", {})
    modules = precision_cfg.get("activation_modules") or precision_cfg.get("modules")
    if not modules:
        modules = list(DEFAULT_PROBE_MODULES)

    num_bits = int(precision_cfg.get("fixed_point_bits", 16))
    fractional_bits = precision_cfg.get("fractional_bits")
    if fractional_bits is not None:
        fractional_bits = int(fractional_bits)
    storage_dtype = _storage_dtype_for_bits(num_bits)
    collector = FixedPointProbeCollector(
        num_bits=num_bits,
        fractional_bits=fractional_bits,
        storage_dtype=storage_dtype,
    )

    for module_name in modules:
        if not hasattr(model, module_name):
            raise ValueError(f"Model does not have module {module_name!r}")
        module = getattr(model, module_name)
        if isinstance(module, FixedPointProbeOutput):
            module = module.module
        setattr(model, module_name, FixedPointProbeOutput(module_name, module, collector))

    model._fixed_point_probe_collector = collector
    model._quantization_summary = {
        "quantization_method": "fixed_point_activation_probe",
        "activation_quantized_modules": list(modules),
        "fixed_point_bits": num_bits,
        "integer_storage_dtype": str(storage_dtype).replace("torch.", ""),
    }
    if fractional_bits is not None:
        model._quantization_summary["fixed_point_fractional_bits"] = fractional_bits
    return model


def quantize_to_fixed_point_integer(
    tensor: torch.Tensor,
    num_bits: int,
    fractional_bits: int | None = None,
    storage_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """Return dequantized values plus the materialized integer tensor."""
    if num_bits < 2:
        raise ValueError("Fixed-point quantization needs at least 2 bits.")
    qmax = (1 << (num_bits - 1)) - 1
    qmin = -(1 << (num_bits - 1))
    if storage_dtype is None:
        storage_dtype = _storage_dtype_for_bits(num_bits)

    if tensor.numel() == 0:
        scale = torch.tensor(1.0, dtype=tensor.dtype, device=tensor.device)
        integer_tensor = torch.empty_like(tensor, dtype=storage_dtype)
        return tensor, integer_tensor, scale, qmin, qmax

    if fractional_bits is None:
        max_abs = tensor.detach().abs().max()
        scale = torch.clamp(max_abs / float(qmax), min=torch.finfo(tensor.dtype).eps)
    else:
        scale = torch.tensor(
            2.0 ** (-fractional_bits),
            dtype=tensor.dtype,
            device=tensor.device,
        )

    rounded = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    integer_tensor = rounded.to(storage_dtype)
    dequantized = integer_tensor.to(torch.float32) * scale.to(torch.float32)
    return dequantized.to(dtype=tensor.dtype), integer_tensor, scale, qmin, qmax


def save_fixed_point_probe_report(model, output_dir: str | Path) -> None:
    """Save per-module fixed-point probe statistics if present on the model."""
    if not hasattr(model, "_fixed_point_probe_collector"):
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = model._fixed_point_probe_collector.as_dict()
    with (output_dir / "fixed_point_probe.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
        file.write("\n")


def _map_floating_tensors(value: Any, fn):
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


def _storage_dtype_for_bits(num_bits: int) -> torch.dtype:
    if num_bits <= 8:
        return torch.int8
    if num_bits <= 16:
        return torch.int16
    if num_bits <= 32:
        return torch.int32
    raise ValueError(f"Unsupported fixed-point width for integer storage: {num_bits}")


def _dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _min_optional(current, value):
    if current is None:
        return value
    return min(current, value)


def _max_optional(current, value):
    if current is None:
        return value
    return max(current, value)
