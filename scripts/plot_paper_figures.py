from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable


MODEL_ORDER = ["bmshj2018-hyperprior", "cheng2020-anchor", "cheng2020-attn"]
MODEL_LABEL = {
    "bmshj2018-hyperprior": "BMShj2018",
    "cheng2020-anchor": "Cheng-anchor",
    "cheng2020-attn": "Cheng-attn",
}

MODEL_STYLE = {
    "bmshj2018-hyperprior": {"color": "#1f77b4", "marker": "o"},
    "cheng2020-anchor": {"color": "#d62728", "marker": "s"},
    "cheng2020-attn": {"color": "#2ca02c", "marker": "^"},
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [{k.lstrip("\ufeff"): v for k, v in row.items()} for row in reader]


def value(row: dict[str, str] | None, key: str) -> float:
    if row is None:
        return math.nan
    raw = row.get(key, "")
    if raw == "":
        return math.nan
    try:
        return float(raw)
    except ValueError:
        return math.nan


def by_model_precision(rows: Iterable[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["model"], row["precision"]): row for row in rows}


def baseline_row(rows: Iterable[dict[str, str]], model: str) -> dict[str, str] | None:
    return next((row for row in rows if row["model"] == model and row["category"] == "baseline"), None)


def configure_matplotlib() -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(
            "Failed to import matplotlib. This usually means the local Python "
            "environment has an incompatible NumPy/matplotlib build. Run this "
            "script on the server environment, or fix the local environment with "
            "`pip install \"numpy<2\" --force-reinstall` or by reinstalling a "
            "matplotlib version compiled for NumPy 2.x.\n"
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
            "lines.linewidth": 1.4,
            "lines.markersize": 4.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        }
    )


def save_figure(fig, out_dir: Path, stem: str, dpi: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi)


def set_zero_line(ax) -> None:
    ax.axhline(0, color="0.55", linewidth=0.8, zorder=0)
    ax.grid(axis="y", color="0.88", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def add_shared_legend(fig, handles, labels, ncol: int = 3) -> None:
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=ncol,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.4,
    )


def plot_precision_sensitivity(rows: list[dict[str, str]], out_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    table = by_model_precision(rows)
    precision_items = [
        ("fp16", "FP16"),
        ("bf16", "BF16"),
        ("int8_ptq_transforms", "INT8-W"),
        ("int8_wa_ptq_calibrated", "INT8-WA"),
        ("int12_fixed_packed", "INT12"),
        ("int10_fixed_packed", "INT10"),
    ]
    x = list(range(len(precision_items)))
    x_labels = [label for _, label in precision_items]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), sharex=True)

    for model in MODEL_ORDER:
        style = MODEL_STYLE[model]
        bpp = []
        psnr = []
        for precision, _ in precision_items:
            row = table.get((model, precision))
            bpp.append(value(row, "bpp_delta_pct"))
            psnr.append(value(row, "psnr_delta"))
        axes[0].plot(x, bpp, label=MODEL_LABEL[model], **style)
        axes[1].plot(x, psnr, label=MODEL_LABEL[model], **style)

    axes[0].set_ylabel(r"$\Delta$ bpp (%)")
    axes[1].set_ylabel(r"$\Delta$ PSNR (dB)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=28, ha="right")
        set_zero_line(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    add_shared_legend(fig, handles, labels)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.5)
    save_figure(fig, out_dir, "fig_precision_sensitivity", dpi)
    plt.close(fig)


def plot_module_sensitivity(rows: list[dict[str, str]], out_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    table = by_model_precision(rows)
    module_items = [
        ("int8_wa_ptq_calibrated_ga", r"$g_a$"),
        ("int8_wa_ptq_calibrated_gs", r"$g_s$"),
        ("int8_wa_ptq_calibrated_hyper", "hyper"),
    ]
    models = MODEL_ORDER
    width = 0.23
    x = list(range(len(module_items)))

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), sharex=True)

    for model_idx, model in enumerate(models):
        offset = (model_idx - 1) * width
        xs = [i + offset for i in x]
        bpp = []
        psnr = []
        for precision, _ in module_items:
            row = table.get((model, precision))
            bpp.append(value(row, "bpp_delta_pct"))
            psnr.append(value(row, "psnr_delta"))
        color = MODEL_STYLE[model]["color"]
        axes[0].bar(xs, bpp, width=width, color=color, label=MODEL_LABEL[model], alpha=0.9)
        valid_xs = [xx for xx, yy in zip(xs, psnr) if math.isfinite(yy)]
        valid_psnr = [yy for yy in psnr if math.isfinite(yy)]
        axes[1].bar(valid_xs, valid_psnr, width=width, color=color, label=MODEL_LABEL[model], alpha=0.9)

    axes[0].set_ylabel(r"$\Delta$ bpp (%)")
    axes[1].set_ylabel(r"$\Delta$ PSNR (dB)")
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in module_items])
        set_zero_line(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    add_shared_legend(fig, handles, labels)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.5)
    save_figure(fig, out_dir, "fig_module_sensitivity", dpi)
    plt.close(fig)


def plot_fixed_point_sensitivity(rows: list[dict[str, str]], out_dir: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    table = by_model_precision(rows)
    bit_items = [
        ("int16_fixed_tensor", "16"),
        ("int12_fixed_packed", "12"),
        ("int10_fixed_packed", "10"),
        ("int8_fixed_packed", "8"),
    ]
    x = [16, 12, 10, 8]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), sharex=True)

    for model in MODEL_ORDER:
        style = MODEL_STYLE[model]
        bpp = []
        psnr = []
        for precision, _ in bit_items:
            row = table.get((model, precision))
            bpp.append(value(row, "bpp_delta_pct"))
            psnr.append(value(row, "psnr_delta"))
        axes[0].plot(x, bpp, label=MODEL_LABEL[model], **style)
        axes[1].plot(x, psnr, label=MODEL_LABEL[model], **style)

    axes[0].set_ylabel(r"$\Delta$ bpp (%)")
    axes[1].set_ylabel(r"$\Delta$ PSNR (dB)")
    for ax in axes:
        ax.set_xlabel("Activation bit width")
        ax.set_xticks(x)
        ax.invert_xaxis()
        set_zero_line(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    add_shared_legend(fig, handles, labels)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=1.5)
    save_figure(fig, out_dir, "fig_fixed_point_sensitivity", dpi)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready figures from summarized codec results."
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("results/tables/main_codec_summary.csv"),
        help="Path to main_codec_summary.csv generated by scripts/summarize_results.py.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("paper/figures"),
        help="Directory for PDF and PNG outputs.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG output resolution.")
    args = parser.parse_args()

    configure_matplotlib()
    rows = read_rows(args.summary)
    plot_precision_sensitivity(rows, args.out_dir, args.dpi)
    plot_module_sensitivity(rows, args.out_dir, args.dpi)
    plot_fixed_point_sensitivity(rows, args.out_dir, args.dpi)

    print(f"Saved figures to {args.out_dir}:")
    print("  fig_precision_sensitivity.pdf/.png")
    print("  fig_module_sensitivity.pdf/.png")
    print("  fig_fixed_point_sensitivity.pdf/.png")


if __name__ == "__main__":
    main()
