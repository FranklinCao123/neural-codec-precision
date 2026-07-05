"""FP16 and mixed-precision inference helpers."""

from __future__ import annotations

import torch

from models.quant_wrappers import AutoCastInput


def apply_fp16_policy(
    model,
    keep_entropy_model_fp32: bool = True,
    modules: list[str] | tuple[str, ...] | None = None,
):
    """Return a model for FP16 experiments.

    FP16 experiments use CUDA autocast around forward and codec calls instead of
    permanently converting the whole model with `model.half()`. This lets CUDA
    kernels run in FP16 where PyTorch considers it valid while preserving FP32
    state such as entropy-model CDF tables.
    """
    if not keep_entropy_model_fp32:
        raise NotImplementedError(
            "Permanent full-model FP16 conversion is not enabled. Use AMP-based "
            "FP16 codec evaluation first; entropy-model state should stay FP32."
        )
    if modules:
        convert_modules_to_fp16(model, modules)
    return model


def convert_modules_to_fp16(model, modules: list[str] | tuple[str, ...]):
    """Convert selected top-level modules, such as g_a/g_s/h_a/h_s, to FP16."""
    for module_name in modules:
        if not hasattr(model, module_name):
            raise ValueError(f"Model does not have module {module_name!r}")
        module = getattr(model, module_name)
        module.half()
        setattr(model, module_name, AutoCastInput(module))
    return model


def model_input_dtype(model) -> torch.dtype:
    """Return the dtype expected by the first learnable parameter."""
    for param in model.parameters():
        return param.dtype
    return torch.float32
