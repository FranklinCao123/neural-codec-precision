"""Generate paper figures from CSV tables produced by summarize_results.py."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable


MODEL_LABEL = {
    "bmshj2018-hyperprior": "BMShj2018",
    "cheng2020-anchor": "Cheng-anchor",
    "cheng2020-attn": "Cheng-attn",
}

PRECISION_LABEL = {
    "fp16": "FP16",
    "bf16": "BF16",
    "int12_fixed_packed": "INT12",
    "int10_fixed_packed": "INT10",
    "int8_wa_ptq_calibrated": "INT8 W+A",
    "int8_ptq_transforms": "INT8 W",
}

PRECISION_STYLE = {
    "fp16": {"color": "#1f77b4", "marker": "o"},
    "bf16": {"color": "#17becf", "marker": "s"},
    "int12_fixed_packed": {"color": "#2ca02c", "marker": "^"},
    "int10_fixed_packed": {"color": "#ff7f0e", "marker": "D"},
    "int8_wa_ptq_calibrated": {"color": "#d62728", "marker": "v"},
    "int8_ptq_transforms": {"color": "#9467bd", "marker": "P"},
}

MODULE_COLOR = {
    "conv": "#4c78a8",
    "attention": "#e45756",
    "residual": "#72b7b2",
    "gdn": "#f58518",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def f(row: dict[str, str] | None, key: str) -> float:
    if row is None:
        return math.nan
    raw = row.get(key, "")
    if raw in {"", "None", "nan", "NaN"}:
        return math.nan
    try:
        return float(raw)
    except ValueError:
        return math.nan


def configure_matplotlib() -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(
            "Failed to import matplotlib. Run this script on the server "
            "environment, or reinstall a NumPy-compatible matplotlib build.\n"
            f"Original error: {exc}"
        ) from exc

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
            "lines.markersize": 4.4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        }
    )


def save_figure(fig, out_dir: Path, stem: str, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi)


def clean_axis(ax, zero_line: bool = True) -> None:
    if zero_line:
        ax.axhline(0, color="0.55", linewidth=0.8, zorder=0)
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def grouped(rows: Iterable[dict[str, str]], *keys: str) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row.get(key, "") for key in keys): row for row in rows}


def plot_quality_trends(rows: list[dict[str, str]], out_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    models = ["cheng2020-anchor", "cheng2020-attn"]
    precisions = ["fp16", "bf16", "int12_fixed_packed", "int10_fixed_packed", "int8_wa_ptq_calibrated"]
    qualities = [1, 3, 5]
    table = grouped(rows, "model", "precision", "quality")

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.4), sharex=True)

    for col, model in enumerate(models):
        for precision in precisions:
            style = PRECISION_STYLE[precision]
            psnr = [f(table.get((model, precision, str(q))), "psnr_delta") for q in qualities]
            bpp = [f(table.get((model, precision, str(q))), "bpp_delta_pct") for q in qualities]
            axes[0, col].plot(qualities, psnr, label=PRECISION_LABEL[precision], **style)
            axes[1, col].plot(qualities, bpp, label=PRECISION_LABEL[precision], **style)

        axes[0, col].set_title(MODEL_LABEL[model], fontsize=8, pad=4)
        axes[1, col].set_xlabel("Quality index")

    axes[0, 0].set_ylabel(r"$\Delta$ PSNR (dB)")
    axes[1, 0].set_ylabel(r"$\Delta$ bpp (%)")
    for ax in axes.ravel():
        ax.set_xticks(qualities)
        clean_axis(ax)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=5,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.4, h_pad=1.0)
    save_figure(fig, out_dir, "fig_quality_precision_trends", dpi)
    plt.close(fig)


def _module_rows(rows: list[dict[str, str]], analysis: str, bits: str = "8") -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("model") == "cheng2020-attn"
        and row.get("quality") in {"1", "3", "5"}
        and row.get("analysis") == analysis
        and (analysis != "quant_error" or row.get("bits") == bits)
        and row.get("module_type") in MODULE_COLOR
    ]


def plot_module_type_sensitivity(rows: list[dict[str, str]], out_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    quant_rows = _module_rows(rows, "quant_error", "8")
    amp_rows = _module_rows(rows, "error_amplification")
    if not quant_rows or not amp_rows:
        return

    module_types = ["conv", "attention", "residual", "gdn"]
    qualities = [1, 3, 5]
    quant = grouped(quant_rows, "quality", "module_type")
    amp = grouped(amp_rows, "quality", "module_type")

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.8), sharex=True)
    width = 0.18
    x = list(range(len(qualities)))
    offsets = {
        module_type: (idx - 1.5) * width
        for idx, module_type in enumerate(module_types)
    }

    for module_type in module_types:
        xs = [value + offsets[module_type] for value in x]
        rel_mse = [f(quant.get((str(q), module_type)), "mean_relative_mse") for q in qualities]
        amplification = [f(amp.get((str(q), module_type)), "mean_error_amplification") for q in qualities]
        axes[0].bar(xs, rel_mse, width=width, color=MODULE_COLOR[module_type], label=module_type)
        axes[1].bar(xs, amplification, width=width, color=MODULE_COLOR[module_type], label=module_type)

    axes[0].set_ylabel("Mean relative MSE")
    axes[1].set_ylabel("Mean error amplification")
    axes[0].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("Quality index")
        ax.set_xticks(x)
        ax.set_xticklabels([str(q) for q in qualities])
        clean_axis(ax, zero_line=False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,
        frameon=False,
        columnspacing=1.2,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.4)
    save_figure(fig, out_dir, "fig_module_type_sensitivity", dpi)
    plt.close(fig)


def plot_layer_ablation(rows: list[dict[str, str]], out_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    if not rows:
        return
    rows = [row for row in rows if row.get("layer") in {"g_a.7", "g_a.8"}]
    table = grouped(rows, "quality", "layer")
    qualities = [1, 3, 5]
    layers = [("g_a.7", r"$g_a.7$ conv"), ("g_a.8", r"$g_a.8$ attention")]
    colors = {"g_a.7": "#4c78a8", "g_a.8": "#e45756"}

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), sharex=True)
    width = 0.32
    x = list(range(len(qualities)))

    for idx, (layer, label) in enumerate(layers):
        offset = (idx - 0.5) * width
        xs = [value + offset for value in x]
        psnr = [f(table.get((str(q), layer)), "psnr_delta") for q in qualities]
        bpp = [f(table.get((str(q), layer)), "bpp_delta_pct") for q in qualities]
        axes[0].bar(xs, psnr, width=width, color=colors[layer], label=label)
        axes[1].bar(xs, bpp, width=width, color=colors[layer], label=label)

    axes[0].set_ylabel(r"$\Delta$ PSNR (dB)")
    axes[1].set_ylabel(r"$\Delta$ bpp (%)")
    for ax in axes:
        ax.set_xlabel("Quality index")
        ax.set_xticks(x)
        ax.set_xticklabels([str(q) for q in qualities])
        clean_axis(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=2,
        frameon=False,
        columnspacing=1.4,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.4)
    save_figure(fig, out_dir, "fig_layer_ablation", dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", type=Path, default=Path("results/tables"), help="Input CSV table directory.")
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"), help="Figure output directory.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG output resolution.")
    args = parser.parse_args()

    configure_matplotlib()
    quality_rows = read_rows(args.tables / "paper_quality_trends.csv")
    module_rows = read_rows(args.tables / "paper_module_type_sensitivity.csv")
    layer_rows = read_rows(args.tables / "paper_layer_ablation.csv")

    plot_quality_trends(quality_rows, args.out_dir, args.dpi)
    plot_module_type_sensitivity(module_rows, args.out_dir, args.dpi)
    plot_layer_ablation(layer_rows, args.out_dir, args.dpi)

    print(f"Saved figures to: {args.out_dir}")
    for path in sorted(args.out_dir.glob("fig_*.*")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
