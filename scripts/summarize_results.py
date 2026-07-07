"""Summarize raw codec experiment results into CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


CORE_COLUMNS = [
    "model",
    "quality",
    "metric",
    "experiment",
    "precision",
    "category",
    "dataset",
    "num_images",
    "avg_bpp",
    "bpp_delta",
    "bpp_delta_pct",
    "avg_psnr",
    "psnr_delta",
    "avg_ms_ssim",
    "ms_ssim_delta",
    "avg_encode_time_sec",
    "encode_time_delta_pct",
    "avg_decode_time_sec",
    "decode_time_delta_pct",
    "avg_forward_time_sec",
    "forward_time_delta_pct",
    "avg_compression_ratio_rgb8",
    "total_bits",
    "num_invalid_reconstructions",
    "total_x_hat_nonfinite_values",
    "param_size_mb",
    "state_dict_size_mb",
    "peak_cuda_memory_mb",
    "quantization_method",
    "quantized_modules",
    "activation_quantized_modules",
    "fixed_point_bits",
    "storage_mode",
]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result


def _is_finite_number(value: Any) -> bool:
    result = _safe_float(value)
    return result is not None and math.isfinite(result)


def _delta(value: Any, baseline: Any) -> float | None:
    value_float = _safe_float(value)
    baseline_float = _safe_float(baseline)
    if value_float is None or baseline_float is None:
        return None
    if not math.isfinite(value_float) or not math.isfinite(baseline_float):
        return None
    return value_float - baseline_float


def _delta_pct(value: Any, baseline: Any) -> float | None:
    value_float = _safe_float(value)
    baseline_float = _safe_float(baseline)
    if value_float is None or baseline_float is None:
        return None
    if not math.isfinite(value_float) or not math.isfinite(baseline_float) or baseline_float == 0.0:
        return None
    return (value_float - baseline_float) / baseline_float * 100.0


def _stringify(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return "|".join(str(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    return value


def _category(row: dict[str, Any]) -> str:
    precision = str(row.get("precision", "")).lower()
    experiment = str(row.get("experiment", "")).lower()
    method = str(row.get("quantization_method", "")).lower()

    if precision in {"fp32", "fp16", "bf16"}:
        return "main_precision"
    if "fp16_weights" in precision:
        return "fp16_weights"
    if "mixed" in precision or "mixed" in experiment:
        return "mixed_precision"
    if "int8_wa_ptq" in precision or "calibrated_ptq" in precision:
        return "int8_wa_ptq"
    if "int8_ptq" in precision:
        return "int8_weight_ptq"
    if "fixed" in precision or "fixed_point" in method:
        return "fixed_point"
    if "fake" in precision:
        return "fake_quant"
    return "other"


def _baseline_key(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return row.get("model"), row.get("quality"), row.get("metric")


def _is_baseline(row: dict[str, Any]) -> bool:
    precision = str(row.get("precision", "")).lower()
    experiment = str(row.get("experiment", "")).lower()
    return precision == "fp32" or "fp32" in experiment


def _read_results(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(input_dir.glob("*/results.json")):
        row = _load_json(path)
        row["result_dir"] = str(path.parent)
        row["category"] = _category(row)
        rows.append(row)
    return rows


def _add_baseline_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = {}
    for row in rows:
        if _is_baseline(row):
            baselines[_baseline_key(row)] = row

    enriched = []
    for row in rows:
        row = dict(row)
        baseline = baselines.get(_baseline_key(row), {})
        row["baseline_experiment"] = baseline.get("experiment")
        row["bpp_delta"] = _delta(row.get("avg_bpp"), baseline.get("avg_bpp"))
        row["bpp_delta_pct"] = _delta_pct(row.get("avg_bpp"), baseline.get("avg_bpp"))
        row["psnr_delta"] = _delta(row.get("avg_psnr"), baseline.get("avg_psnr"))
        row["ms_ssim_delta"] = _delta(row.get("avg_ms_ssim"), baseline.get("avg_ms_ssim"))
        row["encode_time_delta_pct"] = _delta_pct(row.get("avg_encode_time_sec"), baseline.get("avg_encode_time_sec"))
        row["decode_time_delta_pct"] = _delta_pct(row.get("avg_decode_time_sec"), baseline.get("avg_decode_time_sec"))
        row["forward_time_delta_pct"] = _delta_pct(row.get("avg_forward_time_sec"), baseline.get("avg_forward_time_sec"))
        enriched.append(row)
    return enriched


def _invalid_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    invalid = []
    for row in rows:
        has_invalid_recon = int(row.get("num_invalid_reconstructions") or 0) > 0
        has_nonfinite = int(row.get("total_x_hat_nonfinite_values") or 0) > 0
        has_nan_metric = any(
            key in row and not _is_finite_number(row.get(key))
            for key in ["avg_bpp", "avg_psnr", "avg_ms_ssim"]
        )
        if has_invalid_recon or has_nonfinite or has_nan_metric:
            invalid.append(row)
    return invalid


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred_columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_columns = sorted(
        {
            key
            for row in rows
            for key in row
            if key not in preferred_columns
        }
    )
    columns = preferred_columns + extra_columns
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _stringify(row.get(key, "")) for key in columns})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/raw", help="Directory containing */results.json files.")
    parser.add_argument("--output", default="results/tables", help="Output directory for CSV summary tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    rows = _read_results(input_dir)
    if not rows:
        raise FileNotFoundError(f"No results.json files found under {input_dir}")

    rows = _add_baseline_deltas(rows)
    rows = sorted(rows, key=lambda row: (str(row.get("model")), str(row.get("category")), str(row.get("precision"))))

    main_rows = [
        row
        for row in rows
        if row.get("category") in {"main_precision", "int8_weight_ptq", "int8_wa_ptq", "fixed_point", "mixed_precision"}
    ]
    invalid_rows = _invalid_rows(rows)

    _write_csv(output_dir / "all_results.csv", rows, CORE_COLUMNS)
    _write_csv(output_dir / "main_codec_summary.csv", main_rows, CORE_COLUMNS)
    _write_csv(output_dir / "invalid_cases.csv", invalid_rows, CORE_COLUMNS)

    print(f"Read {len(rows)} result files from: {input_dir}")
    print(f"Wrote: {output_dir / 'all_results.csv'}")
    print(f"Wrote: {output_dir / 'main_codec_summary.csv'}")
    print(f"Wrote: {output_dir / 'invalid_cases.csv'}")
    print(f"Invalid/nonfinite cases: {len(invalid_rows)}")


if __name__ == "__main__":
    main()
