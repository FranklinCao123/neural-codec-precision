"""Run INT8 weight fake-quantization PTQ experiments."""

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
from quantization.ptq_int8 import convert_ptq, prepare_ptq


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
    parser.add_argument(
        "--config",
        default="configs/cheng2020_int8_ptq.yaml",
        help="Path to INT8 PTQ experiment config.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to run on: auto, cpu, cuda, cuda:0, etc.",
    )
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
    model = prepare_ptq(model, config)
    model = convert_ptq(model, config)
    dataloader = build_image_dataloader(config)

    results = evaluate_codec(model, dataloader, config, device=device)
    output_dir = PROJECT_ROOT / config.get("output", {}).get("dir", "results/raw/cheng2020_int8_ptq")
    save_results(results, output_dir)

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


if __name__ == "__main__":
    main()
