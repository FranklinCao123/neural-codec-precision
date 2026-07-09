"""Relate module-output quantization to latent changes and actual bitstreams."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.eval_codec import build_image_dataloader
from models.compressai_models import load_model_from_config
from quantization.fake_quant import apply_activation_fake_quant


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to read experiment config files.") from exc
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


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


def _num_bits(value: Any) -> int:
    if isinstance(value, (bytes, bytearray)):
        return len(value) * 8
    if isinstance(value, (list, tuple)):
        return sum(_num_bits(item) for item in value)
    if isinstance(value, dict):
        return sum(_num_bits(item) for item in value.values())
    return 0


def _stream_bits(strings: Any) -> list[int]:
    if not isinstance(strings, (list, tuple)):
        return [_num_bits(strings)]
    return [_num_bits(stream) for stream in strings]


def _relative_mse(reference: torch.Tensor, value: torch.Tensor) -> float:
    error = ((value.float() - reference.float()) ** 2).sum()
    signal = (reference.float() ** 2).sum()
    return float((error / torch.clamp(signal, min=1e-30)).item())


def _rounded_change_ratio(reference: torch.Tensor, value: torch.Tensor) -> float:
    changed = torch.round(reference.float()) != torch.round(value.float())
    return float(changed.float().mean().item())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/cheng2020_attn_fp32.yaml",
        help="FP32 model config used for both the reference and variants.",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=["g_a.7", "g_a.8"],
        help="Qualified module paths whose outputs are quantized independently.",
    )
    parser.add_argument("--bits", type=int, default=8, choices=[8, 10, 12, 16])
    parser.add_argument("--num-images", type=int, default=24)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--output-dir",
        default="results/analysis/cheng2020_attn_latent_changes_q3",
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

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    dataloader = build_image_dataloader(config)
    pad_multiple = int(config.get("evaluation", {}).get("pad_multiple", 64))
    rows: list[dict[str, Any]] = []

    for module_name in args.modules:
        reference_model = load_model_from_config(config).to(device).eval()
        variant_model = copy.deepcopy(reference_model)
        quant_config = {
            "precision": {
                "fake_dtype": f"int{args.bits}",
                "activation_modules": [module_name],
                "fixed_point_bits": args.bits,
            }
        }
        apply_activation_fake_quant(variant_model, quant_config)
        variant_model.eval()

        for index, batch in enumerate(dataloader):
            if index >= args.num_images:
                break
            x = _pad_to_multiple(batch["image"].to(device), pad_multiple)

            y_reference = reference_model.g_a(x)
            y_variant = variant_model.g_a(x)
            z_reference = reference_model.h_a(y_reference)
            z_variant = variant_model.h_a(y_variant)

            reference_compressed = reference_model.compress(x)
            variant_compressed = variant_model.compress(x)
            reference_streams = _stream_bits(reference_compressed["strings"])
            variant_streams = _stream_bits(variant_compressed["strings"])
            reference_total = sum(reference_streams)
            variant_total = sum(variant_streams)

            rows.append(
                {
                    "module": module_name,
                    "bits": args.bits,
                    "image": batch["name"][0],
                    "y_relative_mse": _relative_mse(y_reference, y_variant),
                    "y_rounded_bin_change_ratio": _rounded_change_ratio(y_reference, y_variant),
                    "z_relative_mse": _relative_mse(z_reference, z_variant),
                    "z_rounded_bin_change_ratio": _rounded_change_ratio(z_reference, z_variant),
                    "reference_y_bits": reference_streams[0] if reference_streams else 0,
                    "variant_y_bits": variant_streams[0] if variant_streams else 0,
                    "reference_z_bits": reference_streams[1] if len(reference_streams) > 1 else 0,
                    "variant_z_bits": variant_streams[1] if len(variant_streams) > 1 else 0,
                    "reference_total_bits": reference_total,
                    "variant_total_bits": variant_total,
                    "total_bit_change_ratio": (
                        (variant_total - reference_total) / reference_total
                        if reference_total > 0
                        else 0.0
                    ),
                }
            )

        del variant_model
        del reference_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "per_image.csv", rows)

    summaries = []
    for module_name in args.modules:
        module_rows = [row for row in rows if row["module"] == module_name]
        summary = {"module": module_name, "bits": args.bits, "num_images": len(module_rows)}
        for key in (
            "y_relative_mse",
            "y_rounded_bin_change_ratio",
            "z_relative_mse",
            "z_rounded_bin_change_ratio",
            "total_bit_change_ratio",
        ):
            values = [float(row[key]) for row in module_rows]
            summary[f"mean_{key}"] = sum(values) / max(len(values), 1)
            summary[f"max_{key}"] = max(values, default=0.0)
        reference_total_bits = sum(int(row["reference_total_bits"]) for row in module_rows)
        variant_total_bits = sum(int(row["variant_total_bits"]) for row in module_rows)
        summary["reference_total_bits"] = reference_total_bits
        summary["variant_total_bits"] = variant_total_bits
        summary["aggregate_total_bit_change_ratio"] = (
            (variant_total_bits - reference_total_bits) / reference_total_bits
            if reference_total_bits > 0
            else 0.0
        )
        summaries.append(summary)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "method": "module_output_fake_quant_latent_change_analysis",
                "rounded_bin_change_is_proxy_not_exact_ar_symbol_mismatch": True,
                "config": str(args.config),
                "summaries": summaries,
            },
            file,
            indent=2,
        )
        file.write("\n")

    print(f"Saved latent-change analysis to: {output_dir}")
    for summary in summaries:
        print(
            f"{summary['module']}: "
            f"y_bin_change={summary['mean_y_rounded_bin_change_ratio']:.6f}, "
            f"bit_change={summary['mean_total_bit_change_ratio']:.6f}"
        )


if __name__ == "__main__":
    main()
