"""Calibrated INT8 weight+activation PTQ helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from quantization.ptq_int8 import fake_quantize_modules


DEFAULT_INT8_PTQ_MODULES = ("g_a", "g_s", "h_a", "h_s")


@dataclass
class ActivationObserver:
    module: str
    max_abs: float = 0.0
    num_values: int = 0
    calls: int = 0

    def update(self, value: Any) -> None:
        for tensor in _iter_floating_tensors(value):
            if tensor.numel() == 0:
                continue
            detached = tensor.detach()
            self.max_abs = max(self.max_abs, float(detached.abs().max().item()))
            self.num_values += int(detached.numel())
            self.calls += 1

    def scale(self) -> float:
        if self.max_abs <= 0.0 or not math.isfinite(self.max_abs):
            return 1.0
        return self.max_abs / 127.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "module": self.module,
            "calls": self.calls,
            "num_values": self.num_values,
            "max_abs": self.max_abs,
            "scale": self.scale(),
        }


class CalibratedActivationQuant(nn.Module):
    """Quantize module floating outputs with a fixed calibrated INT8 scale."""

    def __init__(self, module: nn.Module, scale: float):
        super().__init__()
        self.module = module
        self.register_buffer("scale", torch.tensor(float(scale), dtype=torch.float32))

    def forward(self, *args, **kwargs):
        output = self.module(*args, **kwargs)
        return _map_floating_tensors(output, self._quantize)

    def _quantize(self, tensor: torch.Tensor) -> torch.Tensor:
        scale = self.scale.to(device=tensor.device, dtype=tensor.dtype)
        scale = torch.clamp(scale, min=torch.finfo(tensor.dtype).eps)
        quantized = torch.clamp(torch.round(tensor / scale), -127, 127)
        return quantized * scale


def apply_calibrated_int8_ptq(model, config: dict, dataloader, device: torch.device):
    """Calibrate activation scales, then apply INT8 W+A PTQ to selected modules."""
    precision_cfg = config.get("precision", {})
    modules = precision_cfg.get("int8_modules") or precision_cfg.get("activation_modules")
    if not modules:
        modules = list(DEFAULT_INT8_PTQ_MODULES)
    modules = list(modules)

    calibration_cfg = config.get("calibration", {})
    max_images = calibration_cfg.get("num_images")
    max_images = int(max_images) if max_images is not None else None
    pad_multiple = int(config.get("evaluation", {}).get("pad_multiple", 64))

    model = model.to(device)
    model.eval()
    weight_stats = fake_quantize_modules(
        model,
        modules,
        granularity=precision_cfg.get("weight_quantization", "per_channel"),
    )
    observers = calibrate_activation_observers(
        model=model,
        dataloader=dataloader,
        modules=modules,
        device=device,
        max_images=max_images,
        pad_multiple=pad_multiple,
    )

    for module_name in modules:
        module = getattr(model, module_name)
        if isinstance(module, CalibratedActivationQuant):
            module = module.module
        setattr(model, module_name, CalibratedActivationQuant(module, observers[module_name].scale()))

    original_weight_bytes = sum(item.original_bytes for item in weight_stats)
    quantized_weight_bytes = sum(item.simulated_quantized_bytes for item in weight_stats)
    model._quantization_summary = {
        "quantization_method": "int8_weight_activation_calibrated_ptq",
        "quantized_modules": modules,
        "weight_quantization": precision_cfg.get("weight_quantization", "per_channel"),
        "activation_quantization": "calibrated_symmetric_per_tensor",
        "activation_calibration_path": "compress_decompress_after_weight_quantization",
        "calibration_num_images": max_images,
        "activation_observers": [observers[name].as_dict() for name in modules],
        "num_quantized_tensors": len(weight_stats),
        "num_quantized_values": sum(item.numel for item in weight_stats),
        "quantized_weight_original_mb": original_weight_bytes / (1024.0 * 1024.0),
        "quantized_weight_simulated_int8_mb": quantized_weight_bytes / (1024.0 * 1024.0),
        "quantized_weight_size_reduction": (
            1.0 - float(quantized_weight_bytes) / float(original_weight_bytes)
            if original_weight_bytes > 0
            else 0.0
        ),
    }
    return model


@torch.inference_mode()
def calibrate_activation_observers(
    model,
    dataloader,
    modules: list[str],
    device: torch.device,
    max_images: int | None,
    pad_multiple: int,
) -> dict[str, ActivationObserver]:
    observers = {name: ActivationObserver(module=name) for name in modules}
    handles = []

    def make_hook(module_name: str):
        def hook(_module, _inputs, output):
            observers[module_name].update(output)

        return hook

    for module_name in modules:
        if not hasattr(model, module_name):
            raise ValueError(f"Model does not have module {module_name!r}")
        handles.append(getattr(model, module_name).register_forward_hook(make_hook(module_name)))

    try:
        for index, batch in enumerate(dataloader):
            if max_images is not None and index >= max_images:
                break
            x = batch["image"].to(device)
            x_padded = _pad_to_multiple(x, pad_multiple)
            compressed = model.compress(x_padded)
            _ = model.decompress(compressed["strings"], compressed["shape"])
    finally:
        for handle in handles:
            handle.remove()

    return observers


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    _, _, height, width = x.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")


def _iter_floating_tensors(value: Any):
    if torch.is_tensor(value):
        if value.is_floating_point():
            yield value
        return
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_floating_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_floating_tensors(item)


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
