"""Image compression metrics."""


def compute_bpp(num_bits: int, height: int, width: int) -> float:
    """Compute bits per pixel."""
    return float(num_bits) / float(height * width)


def compute_psnr(reference, reconstruction):
    """Compute PSNR for two image tensors."""
    raise NotImplementedError("PSNR computation is not implemented yet.")


def compute_ms_ssim(reference, reconstruction):
    """Compute MS-SSIM for two image tensors."""
    raise NotImplementedError("MS-SSIM computation is not implemented yet.")
