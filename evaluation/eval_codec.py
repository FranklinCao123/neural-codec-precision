"""End-to-end codec evaluation entry points."""

from __future__ import annotations

import json
from contextlib import nullcontext
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from evaluation.metrics import compute_bpp, compute_ms_ssim, compute_psnr


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class ImageFolderDataset(Dataset):
    """A minimal image folder dataset for codec evaluation."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.paths = sorted(
            path
            for path in self.root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in dataset root: {self.root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = _pil_to_tensor(image)

        return {
            "image": tensor,
            "name": path.stem,
            "path": str(path),
        }


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required for image loading.") from exc

    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return tensor


def build_image_dataloader(config: dict) -> DataLoader:
    data_cfg = config.get("data", {})
    dataset = ImageFolderDataset(data_cfg.get("root", "data/kodak"))
    batch_size = int(data_cfg.get("batch_size", 1))
    if batch_size != 1:
        raise ValueError("Real codec evaluation currently requires batch_size=1.")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
    )


def _nested_num_bits(strings: Any) -> int:
    if isinstance(strings, (bytes, bytearray)):
        return len(strings) * 8
    if isinstance(strings, str):
        return len(strings.encode("utf-8")) * 8
    if isinstance(strings, (list, tuple)):
        return sum(_nested_num_bits(item) for item in strings)
    raise TypeError(f"Unsupported bitstream container type: {type(strings)!r}")


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, tuple[int, int]]:
    _, _, height, width = x.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x, (height, width)
    return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate"), (height, width)


def _crop_to_shape(x: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    height, width = shape
    return x[..., :height, :width]


def _save_reconstruction(tensor: torch.Tensor, output_path: Path) -> None:
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError("numpy is required for saving reconstructions.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().cpu().clamp(0.0, 1.0).squeeze(0)
    array = (image.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array).save(output_path)


@torch.inference_mode()
def evaluate_codec(model, dataloader, config: dict, device: str | torch.device = "cpu"):
    """Evaluate bpp, PSNR, MS-SSIM, and timing for a codec model."""
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    eval_cfg = config.get("evaluation", {})
    metrics = set(eval_cfg.get("metrics", ["bpp", "psnr", "ms_ssim"]))
    pad_multiple = int(eval_cfg.get("pad_multiple", 64))
    save_reconstructions = bool(eval_cfg.get("save_reconstructions", False))
    benchmark_forward = bool(eval_cfg.get("benchmark_forward", True))
    forward_warmup = int(eval_cfg.get("forward_warmup", 1))
    forward_repeats = int(eval_cfg.get("forward_repeats", 3))
    forward_precision = eval_cfg.get("forward_precision", config.get("precision", {}).get("mode", "fp32"))
    codec_precision = eval_cfg.get("codec_precision", "fp32")

    output_dir = Path(config.get("output", {}).get("dir", "results/raw/experiment"))
    recon_dir = output_dir / "reconstructions"

    rows = []
    for batch in dataloader:
        x = batch["image"].to(device)
        name = batch["name"][0]
        _, _, height, width = x.shape
        x_padded, original_shape = _pad_to_multiple(x, pad_multiple)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with _autocast_context(device, codec_precision)():
            compressed = model.compress(x_padded)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        encode_time = time.perf_counter() - start

        num_bits = _nested_num_bits(compressed["strings"])

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        with _autocast_context(device, codec_precision)():
            decompressed = model.decompress(compressed["strings"], compressed["shape"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        decode_time = time.perf_counter() - start

        x_hat = _crop_to_shape(decompressed["x_hat"], original_shape)
        x_hat = x_hat.clamp(0.0, 1.0)

        row = {
            "name": name,
            "height": int(height),
            "width": int(width),
            "num_bits": int(num_bits),
            "bpp": compute_bpp(num_bits, height, width),
            "encode_time_sec": encode_time,
            "decode_time_sec": decode_time,
        }
        if benchmark_forward:
            row["forward_time_sec"] = _measure_forward_time(
                model,
                x_padded,
                device,
                warmup=forward_warmup,
                repeats=forward_repeats,
                precision=forward_precision,
            )
        if "psnr" in metrics:
            row["psnr"] = compute_psnr(x, x_hat)
        if "ms_ssim" in metrics or "msssim" in metrics:
            row["ms_ssim"] = compute_ms_ssim(x, x_hat)

        if save_reconstructions:
            _save_reconstruction(x_hat, recon_dir / f"{name}.png")

        rows.append(row)

    summary = _summarize(rows, config)
    return {
        "summary": summary,
        "images": rows,
    }


def _measure_forward_time(
    model,
    x: torch.Tensor,
    device: torch.device,
    warmup: int,
    repeats: int,
    precision: str = "fp32",
) -> float:
    autocast_context = _autocast_context(device, precision)
    for _ in range(max(warmup, 0)):
        with autocast_context():
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()
    for _ in range(max(repeats, 1)):
        with autocast_context():
            _ = model(x)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    return (time.perf_counter() - start) / float(max(repeats, 1))


def _autocast_context(device: torch.device, precision: str):
    precision = precision.lower()
    if precision == "fp16" and device.type == "cuda":
        return lambda: torch.autocast(device_type="cuda", dtype=torch.float16)
    if precision == "bf16" and device.type == "cuda":
        return lambda: torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext


def _summarize(rows: list[dict[str, Any]], config: dict) -> dict[str, Any]:
    def mean(key: str) -> float | None:
        values = [row[key] for row in rows if key in row]
        if not values:
            return None
        return float(sum(values) / len(values))

    model_cfg = config.get("model", {})
    precision_cfg = config.get("precision", {})
    data_cfg = config.get("data", {})

    summary = {
        "experiment": config.get("experiment", {}).get("name"),
        "model": model_cfg.get("name"),
        "quality": model_cfg.get("quality"),
        "metric": model_cfg.get("metric"),
        "precision": precision_cfg.get("mode", "fp32"),
        "dataset": data_cfg.get("dataset"),
        "num_images": len(rows),
        "avg_bpp": mean("bpp"),
        "avg_psnr": mean("psnr"),
        "avg_ms_ssim": mean("ms_ssim"),
        "avg_encode_time_sec": mean("encode_time_sec"),
        "avg_decode_time_sec": mean("decode_time_sec"),
        "avg_forward_time_sec": mean("forward_time_sec"),
        "total_bits": int(sum(row["num_bits"] for row in rows)),
    }
    return {key: value for key, value in summary.items() if value is not None}


def save_results(results: dict[str, Any], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "results.json").open("w", encoding="utf-8") as file:
        json.dump(results["summary"], file, indent=2)
        file.write("\n")

    with (output_dir / "per_image.json").open("w", encoding="utf-8") as file:
        json.dump(results["images"], file, indent=2)
        file.write("\n")
