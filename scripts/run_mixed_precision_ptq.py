"""Run sensitivity-aware mixed precision PTQ experiments."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.eval_codec import build_image_dataloader, evaluate_codec, save_results
from models.compressai_models import load_model_from_config
from quantization.calibrated_int8 import apply_calibrated_int8_ptq
from quantization.fixed_point import apply_fixed_point_probe, save_fixed_point_probe_report
from quantization.fp16 import apply_fp16_policy


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to mixed precision PTQ config.")
    parser.add_argument("--device", default="auto", help="Device to run on: auto, cpu, cuda, etc.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path

    config = _load_yaml(config_path)
    _set_seed(int(config.get("experiment", {}).get("seed", 1234)))

    device = _select_device(args.device)
    model = load_model_from_config(config)
    precision_cfg = config.get("precision", {})
    mixed_cfg = precision_cfg.get("mixed_precision", {})

    fixed_summary = None
    fp16_modules = mixed_cfg.get("fp16_modules") or []
    if fp16_modules:
        model = apply_fp16_policy(
            model,
            keep_entropy_model_fp32=bool(mixed_cfg.get("keep_entropy_model_fp32", True)),
            modules=fp16_modules,
        )

    fixed_modules = mixed_cfg.get("fixed_point_modules") or []
    if fixed_modules:
        fixed_config = _fixed_point_config(config, mixed_cfg, fixed_modules)
        model = apply_fixed_point_probe(model, fixed_config)
        fixed_summary = dict(getattr(model, "_quantization_summary", {}))

    calibration_loader = build_image_dataloader(_calibration_config(config))
    model = apply_calibrated_int8_ptq(model, config, calibration_loader, device=device)
    int8_summary = dict(getattr(model, "_quantization_summary", {}))
    model._quantization_summary = _mixed_summary(
        precision_cfg=precision_cfg,
        mixed_cfg=mixed_cfg,
        fp16_modules=fp16_modules,
        fixed_modules=fixed_modules,
        fixed_summary=fixed_summary,
        int8_summary=int8_summary,
    )

    dataloader = build_image_dataloader(config)
    results = evaluate_codec(model, dataloader, config, device=device)

    output_dir = PROJECT_ROOT / config.get("output", {}).get("dir", "results/raw/mixed_precision_ptq")
    save_results(results, output_dir)
    save_fixed_point_probe_report(model, output_dir)

    summary = results["summary"]
    print(f"Saved results to: {output_dir}")
    print(
        "Summary: "
        f"bpp={summary.get('avg_bpp'):.6f}, "
        f"psnr={summary.get('avg_psnr', float('nan')):.4f}, "
        f"ms_ssim={summary.get('avg_ms_ssim', float('nan')):.6f}, "
        f"enc={summary.get('avg_encode_time_sec'):.4f}s, "
        f"dec={summary.get('avg_decode_time_sec'):.4f}s, "
        f"forward={summary.get('avg_forward_time_sec', float('nan')):.6f}s"
    )


def _calibration_config(config: dict) -> dict:
    calibration_cfg = config.get("calibration", {})
    root = calibration_cfg.get("root") or config.get("data", {}).get("calibration_root")
    if root is None:
        root = config.get("data", {}).get("root", "data/kodak")
    merged = dict(config)
    data_cfg = dict(config.get("data", {}))
    data_cfg["root"] = root
    data_cfg["batch_size"] = 1
    merged["data"] = data_cfg
    return merged


def _fixed_point_config(config: dict, mixed_cfg: dict, fixed_modules: list[str]) -> dict:
    merged = dict(config)
    precision_cfg = dict(config.get("precision", {}))
    precision_cfg["activation_modules"] = list(fixed_modules)
    precision_cfg["fixed_point_bits"] = int(mixed_cfg.get("fixed_point_bits", 16))
    precision_cfg["storage_mode"] = mixed_cfg.get("fixed_point_storage_mode", "tensor")
    precision_cfg["module_fixed_point_bits"] = mixed_cfg.get("module_fixed_point_bits", {})
    if "fractional_bits" in mixed_cfg:
        precision_cfg["fractional_bits"] = mixed_cfg.get("fractional_bits")
    merged["precision"] = precision_cfg
    return merged


def _mixed_summary(
    precision_cfg: dict,
    mixed_cfg: dict,
    fp16_modules: list[str],
    fixed_modules: list[str],
    fixed_summary: dict | None,
    int8_summary: dict,
) -> dict:
    summary = dict(int8_summary)
    summary["quantization_method"] = "mixed_precision_calibrated_ptq"
    summary["mixed_precision_method"] = precision_cfg.get("method")
    summary["fp16_modules"] = list(fp16_modules)
    summary["fixed_point_modules"] = list(fixed_modules)
    summary["int8_modules"] = list(precision_cfg.get("int8_modules", []))
    if fp16_modules:
        summary["fp16_policy"] = "module_weights_fp16_with_fp32_entropy_boundaries"
    if fixed_summary:
        summary["fixed_point_summary"] = fixed_summary
        summary["fixed_point_bits"] = fixed_summary.get("fixed_point_bits")
        summary["fixed_point_storage_mode"] = fixed_summary.get("storage_mode")
    if mixed_cfg:
        summary["mixed_precision_config"] = mixed_cfg
    return summary


if __name__ == "__main__":
    main()
