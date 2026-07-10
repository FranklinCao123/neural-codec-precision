"""Summarize codec and analysis results into paper-ready CSV tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


CORE_COLUMNS = [
    "model",
    "quality",
    "metric",
    "experiment",
    "precision",
    "precision_label",
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
    "avg_decode_time_sec",
    "avg_forward_time_sec",
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

MAIN_PRECISION_LABELS = {
    "fp32": "FP32",
    "fp16": "FP16",
    "bf16": "BF16",
    "int8_ptq_transforms": "INT8-W",
    "int8_wa_ptq_calibrated": "INT8-WA",
    "int16_fixed_tensor": "INT16",
    "int12_fixed_packed": "INT12",
    "int10_fixed_packed": "INT10",
    "int8_fixed_packed": "INT8-fixed",
}

MAIN_PRECISION_ORDER = [
    "fp32",
    "fp16",
    "bf16",
    "int16_fixed_tensor",
    "int12_fixed_packed",
    "int10_fixed_packed",
    "int8_ptq_transforms",
    "int8_wa_ptq_calibrated",
]

MODEL_ORDER = {
    "bmshj2018-hyperprior": 0,
    "cheng2020-anchor": 1,
    "cheng2020-attn": 2,
}


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


def _finite(value: Any) -> bool:
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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred_columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if preferred_columns is None:
        preferred_columns = []
    extra_columns = sorted({key for row in rows for key in row if key not in preferred_columns})
    columns = preferred_columns + extra_columns
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _stringify(row.get(key, "")) for key in columns})


def _category(row: dict[str, Any]) -> str:
    precision = str(row.get("precision", "")).lower()
    experiment = str(row.get("experiment", "")).lower()
    method = str(row.get("quantization_method", "")).lower()

    if precision == "fp32" or "fp32" in experiment:
        return "baseline"
    if precision in {"fp16", "bf16"}:
        return "floating_point"
    if "int8_module" in precision:
        return "layer_ablation"
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


def _precision_label(precision: str) -> str:
    return MAIN_PRECISION_LABELS.get(precision, precision)


def _baseline_key(row: dict[str, Any]) -> tuple[Any, Any, Any]:
    return row.get("model"), row.get("quality"), row.get("metric")


def _read_results(input_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(input_dir.glob("*/results.json")):
        row = _load_json(path)
        row["result_dir"] = str(path.parent)
        row["category"] = _category(row)
        row["precision_label"] = _precision_label(str(row.get("precision", "")))
        rows.append(row)
    return rows


def _add_baseline_deltas(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = {}
    for row in rows:
        if row.get("category") == "baseline":
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
        enriched.append(row)
    return enriched


def _invalid_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    invalid = []
    for row in rows:
        has_invalid_recon = int(row.get("num_invalid_reconstructions") or 0) > 0
        has_nonfinite = int(row.get("total_x_hat_nonfinite_values") or 0) > 0
        has_nan_metric = any(key in row and not _finite(row.get(key)) for key in ["avg_bpp", "avg_psnr", "avg_ms_ssim"])
        if has_invalid_recon or has_nonfinite or has_nan_metric:
            invalid.append(row)
    return invalid


def _model_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    precision = str(row.get("precision", ""))
    precision_idx = MAIN_PRECISION_ORDER.index(precision) if precision in MAIN_PRECISION_ORDER else 999
    return (
        MODEL_ORDER.get(str(row.get("model")), 999),
        int(row.get("quality") or 0),
        precision_idx,
        str(row.get("experiment")),
    )


def _main_codec_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    categories = {"baseline", "floating_point", "int8_weight_ptq", "int8_wa_ptq", "fixed_point", "mixed_precision"}
    return [row for row in rows if row.get("category") in categories]


def _paper_q3_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if int(row.get("quality") or 0) != 3:
            continue
        if str(row.get("precision")) not in MAIN_PRECISION_ORDER:
            continue
        output.append(row)
    return sorted(output, key=_model_sort_key)


def _paper_quality_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    wanted_models = {"cheng2020-anchor", "cheng2020-attn"}
    wanted_qualities = {1, 3, 5}
    for row in rows:
        if row.get("model") not in wanted_models:
            continue
        if int(row.get("quality") or 0) not in wanted_qualities:
            continue
        if str(row.get("precision")) not in MAIN_PRECISION_ORDER:
            continue
        output.append(row)
    return sorted(output, key=_model_sort_key)


def _layer_name(row: dict[str, Any]) -> str:
    modules = row.get("activation_quantized_modules") or []
    if isinstance(modules, list) and modules:
        return str(modules[0])
    experiment = str(row.get("experiment", ""))
    match = re.search(r"module_(ga7|ga8|ga3|gs0|gs5)", experiment)
    return match.group(1) if match else ""


def _layer_type(layer: str) -> str:
    mapping = {
        "g_a.7": "conv",
        "g_a.8": "attention",
        "g_a.3": "attention",
        "g_s.0": "attention",
        "g_s.5": "attention",
        "ga7": "conv",
        "ga8": "attention",
    }
    return mapping.get(layer, "")


def _paper_layer_ablation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if row.get("category") != "layer_ablation":
            continue
        experiment = str(row.get("experiment", ""))
        if "p999" in experiment:
            continue
        layer = _layer_name(row)
        if layer not in {"g_a.7", "g_a.8", "ga7", "ga8"}:
            continue
        row = dict(row)
        row["layer"] = layer
        row["layer_type"] = _layer_type(layer)
        output.append(row)
    return sorted(output, key=lambda row: (int(row.get("quality") or 0), row.get("layer")))


def _paper_storage_rows(rows: list[dict[str, Any]], raw_dir: Path) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        if row.get("category") != "fixed_point":
            continue
        probe_path = raw_dir / Path(str(row.get("result_dir", ""))).name / "fixed_point_probe.json"
        probe = _load_json(probe_path) if probe_path.exists() else {}
        storage = {
            "packed_storage_reduction": probe.get("total_packed_storage_reduction"),
            "tensor_storage_reduction": probe.get("total_storage_reduction"),
            "total_saturation_ratio": probe.get("total_saturation_ratio"),
            "total_pack_time_sec": probe.get("total_pack_time_sec"),
            "total_unpack_time_sec": probe.get("total_unpack_time_sec"),
        }
        output.append({**row, **storage})
    return sorted(output, key=_model_sort_key)


def _quality_from_analysis_dir(name: str) -> int | None:
    match = re.search(r"_q([135])$", name)
    if match:
        return int(match.group(1))
    if name == "cheng2020_attn":
        return 3
    return None


def _paper_module_type_rows(analysis_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for model_dir in sorted(analysis_dir.glob("*")):
        if not model_dir.is_dir():
            continue
        quality = _quality_from_analysis_dir(model_dir.name)
        model = "cheng2020-attn" if "cheng2020_attn" in model_dir.name else model_dir.name

        bits_path = model_dir / "summary_by_type_bits.csv"
        if bits_path.exists():
            for row in _read_csv(bits_path):
                row = dict(row)
                row["model"] = model
                row["quality"] = quality
                row["analysis"] = "quant_error"
                rows.append(row)

        amp_path = model_dir / "error_amplification_by_type.csv"
        if amp_path.exists():
            for row in _read_csv(amp_path):
                row = dict(row)
                row["model"] = model
                row["quality"] = quality
                row["analysis"] = "error_amplification"
                rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="results/raw", help="Directory containing */results.json files.")
    parser.add_argument("--analysis", default="results/analysis", help="Directory containing module analysis outputs.")
    parser.add_argument("--output", default="results/tables", help="Output directory for CSV summary tables.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    analysis_dir = Path(args.analysis)
    output_dir = Path(args.output)

    rows = _read_results(input_dir)
    if not rows:
        raise FileNotFoundError(f"No results.json files found under {input_dir}")

    rows = _add_baseline_deltas(rows)
    rows = sorted(rows, key=_model_sort_key)
    main_rows = _main_codec_rows(rows)
    invalid_rows = _invalid_rows(rows)
    q3_rows = _paper_q3_rows(main_rows)
    quality_rows = _paper_quality_rows(main_rows)
    layer_rows = _paper_layer_ablation_rows(rows)
    storage_rows = _paper_storage_rows(rows, input_dir)
    module_type_rows = _paper_module_type_rows(analysis_dir)

    _write_csv(output_dir / "all_results.csv", rows, CORE_COLUMNS)
    _write_csv(output_dir / "main_codec_summary.csv", main_rows, CORE_COLUMNS)
    _write_csv(output_dir / "invalid_cases.csv", invalid_rows, CORE_COLUMNS)
    _write_csv(output_dir / "paper_main_q3.csv", q3_rows, CORE_COLUMNS)
    _write_csv(output_dir / "paper_quality_trends.csv", quality_rows, CORE_COLUMNS)
    _write_csv(output_dir / "paper_layer_ablation.csv", layer_rows, CORE_COLUMNS + ["layer", "layer_type"])
    _write_csv(
        output_dir / "paper_fixed_storage.csv",
        storage_rows,
        CORE_COLUMNS
        + [
            "tensor_storage_reduction",
            "packed_storage_reduction",
            "total_saturation_ratio",
            "total_pack_time_sec",
            "total_unpack_time_sec",
        ],
    )
    _write_csv(output_dir / "paper_module_type_sensitivity.csv", module_type_rows)

    print(f"Read {len(rows)} result files from: {input_dir}")
    print(f"Wrote tables to: {output_dir}")
    print(f"Main codec rows: {len(main_rows)}")
    print(f"q3 paper rows: {len(q3_rows)}")
    print(f"quality trend rows: {len(quality_rows)}")
    print(f"layer ablation rows: {len(layer_rows)}")
    print(f"module type analysis rows: {len(module_type_rows)}")
    print(f"Invalid/nonfinite cases: {len(invalid_rows)}")


if __name__ == "__main__":
    main()
