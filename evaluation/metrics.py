"""Image compression metrics."""

from __future__ import annotations

import math

import torch


def compute_bpp(num_bits: int, height: int, width: int) -> float:
    """Compute bits per pixel."""
    return float(num_bits) / float(height * width)


def compute_psnr(reference, reconstruction) -> float:
    """Compute PSNR for image tensors in [0, 1]."""
    ref = reference.detach().float()
    rec = reconstruction.detach().float().clamp(0.0, 1.0)
    mse = torch.mean((ref - rec) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def compute_ms_ssim(reference, reconstruction) -> float:
    """Compute MS-SSIM for image tensors in [0, 1]."""
    try:
        from pytorch_msssim import ms_ssim
    except ImportError as exc:
        raise ImportError(
            "pytorch-msssim is required for MS-SSIM. Install it on the server "
            "or remove ms_ssim from the config metrics list."
        ) from exc

    ref = reference.detach().float()
    rec = reconstruction.detach().float().clamp(0.0, 1.0)
    return float(ms_ssim(rec, ref, data_range=1.0).item())
