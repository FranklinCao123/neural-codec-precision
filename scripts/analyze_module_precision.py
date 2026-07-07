"""Analyze module activation distributions and low-precision sensitivity."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.eval_codec import build_image_dataloader
from models.compressai_models import load_model_from_config


DEFAULT_BITS = (8, 10, 12, 16)
TOP_LEVEL_MODULES = {"g_a", "g_s", "h_a", "h_s"}


class ActivationStats:
    def __init__(self) -> None:
        self.calls = 0
        self.num_values = 0
        self.sum_values = 0.0
        self.sum_squares = 0.0
        self.min_value = math.inf
        self.max_value = -math.inf
        self.max_abs = 0.0
        self.sum_abs_mean = 0.0
        self.sum_std = 0.0
        self.sum_p99_abs = 0.0
        self.max_p99_abs = 0.0
        self.sum_p999_abs = 0.0
        self.max_p999_abs = 0.0

    def update(self, tensor: torch.Tensor) -> None:
        values = tensor.detach().float()
        if values.numel() == 0:
            return

        flat = values.reshape(-1)
        abs_flat = flat.abs()
        num_values = int(flat.numel())
        self.calls += 1
        self.num_values += num_values
        self.sum_values += float(flat.sum().item())
        self.sum_squares += float((flat * flat).sum().item())
        self.min_value = min(self.min_value, float(flat.min().item()))
        self.max_value = max(self.max_value, float(flat.max().item()))
        self.max_abs = max(self.max_abs, float(abs_flat.max().item()))
        self.sum_abs_mean += float(abs_flat.mean().item())
        self.sum_std += float(flat.std(unbiased=False).item())

        p99_abs = _quantile(abs_flat, 0.99)
        p999_abs = _quantile(abs_flat, 0.999)
        self.sum_p99_abs += p99_abs
        self.max_p99_abs = max(self.max_p99_abs, p99_abs)
        self.sum_p999_abs += p999_abs
        self.max_p999_abs = max(self.max_p999_abs, p999_abs)

    def to_row(self, module: dict[str, str]) -> dict[str, Any]:
        mean = self.sum_values / max(self.num_values, 1)
        variance = self.sum_squares / max(self.num_values, 1) - mean * mean
        variance = max(variance, 0.0)
        avg_p999_abs = self.sum_p999_abs / max(self.calls, 1)
        max_abs = self.max_abs
        return {
            **module,
            "calls": self.calls,
            "num_values": self.num_values,
            "min": self.min_value,
            "max": self.max_value,
            "mean": mean,
            "std": math.sqrt(variance),
            "avg_call_std": self.sum_std / max(self.calls, 1),
            "avg_abs_mean": self.sum_abs_mean / max(self.calls, 1),
            "max_abs": max_abs,
            "avg_p99_abs": self.sum_p99_abs / max(self.calls, 1),
            "max_p99_abs": self.max_p99_abs,
            "avg_p999_abs": avg_p999_abs,
            "max_p999_abs": self.max_p999_abs,
            "p999_to_max_abs": avg_p999_abs / max(max_abs, 1e-12),
            "int8_dynamic_scale": max_abs / 127.0 if max_abs > 0 else 1.0,
        }


class QuantErrorStats:
    def __init__(self, bits: int) -> None:
        self.bits = bits
        self.calls = 0
        self.num_values = 0
        self.sse = 0.0
        self.signal = 0.0
        self.dot = 0.0
        self.dequant_signal = 0.0
        self.max_abs_error = 0.0
        self.sum_scale = 0.0
        self.max_scale = 0.0
        self.saturated_values = 0

    def update(self, tensor: torch.Tensor) -> None:
        values = tensor.detach().float()
        if values.numel() == 0:
            return

        dequantized, quantized, scale, qmin, qmax = _fake_quantize(values, self.bits)
        error = dequantized - values
        num_values = int(values.numel())

        self.calls += 1
        self.num_values += num_values
        self.sse += float((error * error).sum().item())
        self.signal += float((values * values).sum().item())
        self.dot += float((values * dequantized).sum().item())
        self.dequant_signal += float((dequantized * dequantized).sum().item())
        self.max_abs_error = max(self.max_abs_error, float(error.abs().max().item()))
        self.sum_scale += float(scale)
        self.max_scale = max(self.max_scale, float(scale))
        self.saturated_values += int(((quantized == qmin) | (quantized == qmax)).sum().item())

    def to_row(self, module: dict[str, str]) -> dict[str, Any]:
        mse = self.sse / max(self.num_values, 1)
        signal_power = self.signal / max(self.num_values, 1)
        relative_mse = self.sse / max(self.signal, 1e-30)
        sqnr_db = 10.0 * math.log10(max(self.signal, 1e-30) / max(self.sse, 1e-30))
        cosine = self.dot / math.sqrt(max(self.signal * self.dequant_signal, 1e-30))
        return {
            **module,
            "bits": self.bits,
            "calls": self.calls,
            "num_values": self.num_values,
            "mse": mse,
            "signal_power": signal_power,
            "relative_mse": relative_mse,
            "sqnr_db": sqnr_db,
            "cosine_similarity": cosine,
            "max_abs_error": self.max_abs_error,
            "avg_scale": self.sum_scale / max(self.calls, 1),
            "max_scale": self.max_scale,
            "saturated_values": self.saturated_values,
            "saturation_ratio": self.saturated_values / max(self.num_values, 1),
        }


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to read experiment config files.") from exc

    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _pad_to_multiple(x: torch.Tensor, multiple: int) -> torch.Tensor:
    _, _, height, width = x.shape
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")


def _module_type(module: nn.Module) -> str | None:
    class_name = module.__class__.__name__
    if "Attention" in class_name:
        return "attention"
    if "GDN" in class_name:
        return "gdn"
    if "ResidualBlock" in class_name:
        return "residual"
    if isinstance(module, nn.ConvTranspose2d):
        return "deconv"
    if isinstance(module, nn.Conv2d):
        return "conv"
    return None


def _top_level_name(name: str) -> str:
    return name.split(".", 1)[0] if name else ""


def _select_modules(model: nn.Module, include_other_top_levels: bool) -> list[dict[str, Any]]:
    selected = []
    for name, module in model.named_modules():
        if not name:
            continue
        module_type = _module_type(module)
        top_level = _top_level_name(name)
        if module_type is None:
            continue
        if not include_other_top_levels and top_level not in TOP_LEVEL_MODULES:
            continue
        selected.append(
            {
                "name": name,
                "class_name": module.__class__.__name__,
                "module_type": module_type,
                "top_level": top_level,
                "module": module,
            }
        )
    return selected


def _first_tensor(output: Any) -> torch.Tensor | None:
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for value in output.values():
            tensor = _first_tensor(value)
            if tensor is not None:
                return tensor
    if isinstance(output, (list, tuple)):
        for value in output:
            tensor = _first_tensor(value)
            if tensor is not None:
                return tensor
    return None


def _quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 1:
        return float(values.item())
    return float(torch.quantile(values, q).item())


def _fake_quantize(values: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor, float, int, int]:
    qmax = (1 << (bits - 1)) - 1
    qmin = -(1 << (bits - 1))
    max_abs = float(values.abs().max().item())
    scale = max_abs / float(qmax) if max_abs > 0.0 else 1.0
    quantized = torch.round(values / scale).clamp(qmin, qmax)
    dequantized = quantized * scale
    return dequantized, quantized, scale, qmin, qmax


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)
        file.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as file:
            file.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summarize_numeric(rows: list[dict[str, Any]], group_key: str, metric_keys: list[str]) -> list[dict[str, Any]]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_key]].append(row)

    summary_rows = []
    for group, group_rows in sorted(grouped.items()):
        summary = {
            group_key: group,
            "num_modules": len(group_rows),
            "total_values": int(sum(int(row.get("num_values", 0)) for row in group_rows)),
        }
        for key in metric_keys:
            values = [float(row[key]) for row in group_rows if key in row and row[key] != ""]
            if values:
                summary[f"mean_{key}"] = sum(values) / len(values)
                summary[f"max_{key}"] = max(values)
        summary_rows.append(summary)
    return summary_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to an FP32 model config.")
    parser.add_argument("--device", default="auto", help="Device to run on: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--num-images", type=int, default=4, help="Number of images used for the analysis.")
    parser.add_argument(
        "--bits",
        nargs="+",
        type=int,
        default=list(DEFAULT_BITS),
        help="Activation quantization bit widths to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for modules.json and CSV analysis outputs.",
    )
    parser.add_argument(
        "--include-other-top-levels",
        action="store_true",
        help="Also analyze selected module types outside g_a/g_s/h_a/h_s.",
    )
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config = _load_yaml(config_path)
    _set_seed(int(config.get("experiment", {}).get("seed", 1234)))
    device = _select_device(args.device)
    model = load_model_from_config(config).to(device)
    model.eval()

    modules = _select_modules(model, include_other_top_levels=args.include_other_top_levels)
    if not modules:
        raise RuntimeError("No Conv/GDN/Residual/Attention modules were selected for analysis.")

    module_by_name = {
        item["name"]: {
            "name": item["name"],
            "class_name": item["class_name"],
            "module_type": item["module_type"],
            "top_level": item["top_level"],
        }
        for item in modules
    }
    activation_stats = {item["name"]: ActivationStats() for item in modules}
    quant_stats = {
        item["name"]: {bits: QuantErrorStats(bits) for bits in args.bits}
        for item in modules
    }

    hooks = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            tensor = _first_tensor(output)
            if tensor is None or not torch.is_floating_point(tensor):
                return
            activation_stats[name].update(tensor)
            for stats in quant_stats[name].values():
                stats.update(tensor)

        return hook

    for item in modules:
        hooks.append(item["module"].register_forward_hook(make_hook(item["name"])))

    dataloader = build_image_dataloader(config)
    pad_multiple = int(config.get("evaluation", {}).get("pad_multiple", 64))
    num_images = 0
    try:
        for batch in dataloader:
            x = batch["image"].to(device)
            x = _pad_to_multiple(x, pad_multiple)
            _ = model(x)
            num_images += 1
            if num_images >= args.num_images:
                break
    finally:
        for hook in hooks:
            hook.remove()

    model_name = str(config.get("model", {}).get("name", "model")).replace("/", "_")
    if args.output_dir is None:
        output_dir = PROJECT_ROOT / "results" / "analysis" / model_name
    else:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir

    module_rows = list(module_by_name.values())
    activation_rows = [
        activation_stats[name].to_row(module_by_name[name])
        for name in sorted(module_by_name)
        if activation_stats[name].calls > 0
    ]
    quant_rows = [
        quant_stats[name][bits].to_row(module_by_name[name])
        for name in sorted(module_by_name)
        for bits in sorted(quant_stats[name])
        if quant_stats[name][bits].calls > 0
    ]

    _write_json(
        output_dir / "modules.json",
        {
            "config": str(config_path.relative_to(PROJECT_ROOT) if config_path.is_relative_to(PROJECT_ROOT) else config_path),
            "model": config.get("model", {}),
            "num_images": num_images,
            "bits": args.bits,
            "modules": module_rows,
        },
    )
    _write_csv(output_dir / "activation_stats.csv", activation_rows)
    _write_csv(output_dir / "quant_error.csv", quant_rows)
    _write_csv(
        output_dir / "summary_by_type.csv",
        _summarize_numeric(
            quant_rows,
            group_key="module_type",
            metric_keys=["relative_mse", "sqnr_db", "cosine_similarity", "saturation_ratio"],
        ),
    )
    _write_csv(
        output_dir / "summary_by_top_level.csv",
        _summarize_numeric(
            quant_rows,
            group_key="top_level",
            metric_keys=["relative_mse", "sqnr_db", "cosine_similarity", "saturation_ratio"],
        ),
    )

    print(f"Saved module precision analysis to: {output_dir}")
    print(f"Analyzed {len(activation_rows)} modules on {num_images} images.")


if __name__ == "__main__":
    main()
