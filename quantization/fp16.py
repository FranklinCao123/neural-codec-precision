"""FP16 and mixed-precision inference helpers."""

from __future__ import annotations


def apply_fp16_policy(model, keep_entropy_model_fp32: bool = True):
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
    return model
