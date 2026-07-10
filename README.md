# Neural Codec Precision Experiments

This repository contains experiments for studying low-precision inference,
post-training quantization, and fixed-point activation storage in learned image
compression models.

The main goal is to evaluate how different numerical precisions affect:

- rate-distortion performance: bpp, PSNR, MS-SSIM
- runtime behavior: encode/decode time, forward time, CUDA memory
- model footprint: parameter/state-dict size and simulated quantized weight size
- stability: invalid reconstructions and non-finite decoded values
- module sensitivity: encoder, decoder, hyperprior, GDN, residual, and attention blocks

The paper title used for the current report is:

```text
Post-Training Quantization and Fixed-Point Analysis for Learned Image Compression
```

## Models

The current experiments use pretrained CompressAI models:

| Config prefix | CompressAI model | Role |
| --- | --- | --- |
| `bmshj2018_hyperprior_*` | `bmshj2018-hyperprior` | CNN hyperprior baseline |
| `cheng2020_*` | `cheng2020-anchor` | CNN + hyperprior + autoregressive context |
| `cheng2020_attn_*` | `cheng2020-attn` | Cheng model with attention modules |

The main paper results use Kodak images and `metric=mse`. BMShj2018-hyperprior
is evaluated at `quality=3`; Cheng2020-anchor and Cheng2020-attn are evaluated
at `quality=1,3,5` for quality-level trend analysis.

## Experiment Types

The project covers these precision settings:

- `fp32`: full-precision codec baseline
- `fp16`, `bf16`: low-precision floating-point inference
- `fp16_weights_*`: module-level FP16 weight experiments
- `int8_ptq_*`: INT8 weight-only post-training quantization
- `int8_wa_ptq_calibrated_*`: calibrated INT8 weight + activation PTQ
- `int{8,10,12,16}_fixed_*`: fixed-point activation quantize-store-dequantize experiments
- `mixed_*`: mixed-precision module policies
- `fp8_fake_*`, `int*_fake_*`: exploratory fake-quantization experiments

Important scope note: INT/fixed-point experiments preserve the CompressAI entropy
coder and probability table machinery in floating point where needed. This keeps
full `compress()` / `decompress()` evaluation valid while isolating neural-module
precision effects.

The calibrated INT8 W+A PTQ experiments use an independent calibration set
(`data/calibration`, CBSD20 in the current report) to estimate activation
ranges. Kodak images are used only for evaluation.

## Repository Layout

```text
configs/        YAML experiment configs.
data/           Local datasets. Ignored by git.
docs/           Notes and experiment plans.
evaluation/     Codec evaluation, metrics, timing, and size utilities.
models/         CompressAI model loading and wrappers.
paper/          Local paper sources and generated paper figures. Ignored by git.
quantization/   FP16, PTQ, calibrated INT8, QAT, and fixed-point utilities.
scripts/        Command-line entry points.
results/        Generated experiment outputs. Ignored by git except placeholders.
```

## Setup

Install the main dependencies on the server:

```bash
pip install compressai pytorch-msssim pyyaml pillow numpy
```

Install PyTorch according to the CUDA version of the server before running the
experiments.

Expected dataset layout:

```text
data/kodak/
  kodim01.png
  ...
  kodim24.png

data/calibration/
  0001.png
  ...
```

For visualization-only runs:

```bash
mkdir -p data/kodak_viz
cp data/kodak/kodim01.png data/kodak_viz/
cp data/kodak/kodim19.png data/kodak_viz/
```

## Common Commands

Run a baseline:

```bash
python scripts/run_fp32_baseline.py --config configs/cheng2020_fp32.yaml --device cuda
```

For quality-level Cheng experiments, use configs such as:

```bash
python scripts/run_fp32_baseline.py --config configs/cheng2020_anchor_q1_fp32.yaml --device cuda
python scripts/run_fp32_baseline.py --config configs/cheng2020_anchor_q5_fp32.yaml --device cuda
python scripts/run_fp32_baseline.py --config configs/cheng2020_attn_q1_fp32.yaml --device cuda
python scripts/run_fp32_baseline.py --config configs/cheng2020_attn_q5_fp32.yaml --device cuda
```

Run FP16 and BF16:

```bash
python scripts/run_fp16.py --config configs/cheng2020_fp16.yaml --device cuda
python scripts/run_bf16.py --config configs/cheng2020_bf16.yaml --device cuda
```

Run INT8 weight-only PTQ:

```bash
python scripts/run_int8_ptq.py --config configs/cheng2020_int8_ptq_transforms.yaml --device cuda
```

Run calibrated INT8 weight + activation PTQ:

```bash
python scripts/run_int8_calibrated_ptq.py \
  --config configs/cheng2020_int8_wa_ptq_calibrated.yaml \
  --device cuda
```

The calibrated PTQ config should point to `data/calibration` for calibration
and to `data/kodak` for evaluation.

Run fixed-point activation codec experiments:

```bash
python scripts/run_fixed_point_codec.py --config configs/cheng2020_int16_fixed_tensor.yaml --device cuda
python scripts/run_fixed_point_codec.py --config configs/cheng2020_int12_fixed_packed.yaml --device cuda
python scripts/run_fixed_point_codec.py --config configs/cheng2020_int10_fixed_packed.yaml --device cuda
python scripts/run_fixed_point_codec.py --config configs/cheng2020_int8_fixed_packed.yaml --device cuda
```

## Recommended Experiment Matrix

For each main model, the core set is:

```text
FP32
FP16
BF16
INT8 weight-only PTQ
INT8 weight + activation calibrated PTQ
INT16 fixed-point
INT12 fixed-point
INT10 fixed-point
INT8 fixed-point when stable or useful
```

For the current report, the minimum matrix is:

```text
BMShj2018-hyperprior q3:
  FP32, FP16, BF16, INT8 W-only PTQ, INT8 W+A calibrated PTQ,
  INT16 fixed, INT12 fixed, INT10 fixed

Cheng2020-anchor q1/q3/q5:
  FP32, FP16, BF16, INT8 W-only PTQ, INT8 W+A calibrated PTQ,
  INT12 fixed, INT10 fixed

Cheng2020-attn q1/q3/q5:
  FP32, FP16, BF16, INT8 W-only PTQ, INT8 W+A calibrated PTQ,
  INT12 fixed, INT10 fixed
```

Module-wise INT8 W+A PTQ ablations use configs such as:

```text
*_int8_wa_ptq_calibrated_ga.yaml
*_int8_wa_ptq_calibrated_gs.yaml
*_int8_wa_ptq_calibrated_hyper.yaml
*_int8_wa_ptq_calibrated_gs_hyper.yaml
```

These experiments help separate rate sensitivity from reconstruction sensitivity:

- `g_a`: analysis transform / encoder; often affects latent distribution and bpp
- `g_s`: synthesis transform / decoder; often affects PSNR/MS-SSIM
- `h_a`, `h_s`: hyperprior path; affects entropy-parameter prediction and bpp

## Module Precision Analysis

Use this script to collect activation statistics and quantization error by
module type and top-level path:

```bash
python scripts/analyze_module_precision.py \
  --config configs/cheng2020_attn_fp32.yaml \
  --device cuda \
  --num-images 4 \
  --output-dir results/analysis/cheng2020_attn
```

Outputs:

```text
modules.json
activation_stats.csv
quant_error.csv
summary_by_type.csv
summary_by_type_bits.csv
summary_by_top_level.csv
summary_by_top_level_bits.csv
worst_int8_layers.csv
```

The per-bit summaries are the main files for comparing Conv, GDN, residual, and
attention sensitivity.

For cause analysis around attention and convolution sensitivity, run the
quality-specific analysis configs:

```bash
python scripts/analyze_module_precision.py \
  --config configs/cheng2020_attn_q1_fp32.yaml \
  --device cuda --num-images 8 --bits 8 12 --amplification-bits 8 \
  --output-dir results/analysis/cheng2020_attn_cause_q1

python scripts/analyze_module_precision.py \
  --config configs/cheng2020_attn_q5_fp32.yaml \
  --device cuda --num-images 8 --bits 8 12 --amplification-bits 8 \
  --output-dir results/analysis/cheng2020_attn_cause_q5
```

## Visualization

Visualization runs save reconstructions for a small Kodak subset:

```bash
python scripts/run_fp32_baseline.py --config configs/viz_cheng2020_attn_fp32.yaml --device cuda
python scripts/run_fp16.py --config configs/viz_cheng2020_attn_fp16.yaml --device cuda
python scripts/run_fixed_point_codec.py --config configs/viz_cheng2020_attn_int12_fixed_packed.yaml --device cuda
python scripts/run_int8_calibrated_ptq.py --config configs/viz_cheng2020_attn_int8_wa_ptq_calibrated.yaml --device cuda
```

Create a cropped comparison figure:

```bash
python scripts/make_visual_comparison.py \
  --item Original=data/kodak_viz/kodim01.png \
  --item FP32=results/visualizations/cheng2020_attn_fp32/reconstructions/kodim01.png \
  --item FP16=results/visualizations/cheng2020_attn_fp16/reconstructions/kodim01.png \
  --item INT12=results/visualizations/cheng2020_attn_int12_fixed_packed/reconstructions/kodim01.png \
  --item INT8=results/visualizations/cheng2020_attn_int8_wa_ptq_calibrated/reconstructions/kodim01.png \
  --crop 250,150,160,160 \
  --output results/visualizations/figures/kodim01_cheng2020_attn_crop.png
```

## Result Summaries

After downloading or generating `results/raw/*/results.json`, summarize all
experiments with:

```bash
python scripts/summarize_results.py --input results/raw --output results/tables
```

Generated tables:

```text
results/tables/all_results.csv
results/tables/main_codec_summary.csv
results/tables/invalid_cases.csv
```

Paper-specific tables and figures are generated from the downloaded or
generated results:

```bash
python scripts/build_result_tables.py
python scripts/plot_paper_figures.py
```

Current paper tables:

```text
results/tables/paper_main_q3.csv
results/tables/paper_quality_trends.csv
results/tables/paper_fixed_storage_compact.csv
results/tables/paper_module_type_sensitivity.csv
results/tables/paper_layer_ablation.csv
```

Current paper figures:

```text
results/figures/fig_quality_precision_trends.pdf
results/figures/fig_module_type_sensitivity.pdf
results/figures/fig_layer_ablation.pdf
```

Useful checks:

```bash
find results/raw -maxdepth 2 -name results.json | sort
find results/analysis -maxdepth 2 -type f | sort
find results/visualizations -type f | sort
cat results/tables/invalid_cases.csv
```

## Paper

The paper is kept under `paper/` for local writing and is ignored by git. The
English source uses the figures in `paper/figures/` and the bibliography file
`paper/references.bib`.

```text
paper/conference_101719.tex
paper/references.bib
paper/figures/
```

A standalone Chinese LaTeX project is available under:

```text
paper/chineses/
  conference_101719_zh.tex
  references.bib
  figures/
```

The Chinese project is self-contained and can be zipped directly for Overleaf.
Compile the Chinese version with XeLaTeX because it uses `xeCJK`.

## Git Policy

Generated data and experiment outputs are ignored:

```text
data/*
checkpoints/*
results/raw/*
results/analysis/*
results/visualizations/*
results/tables/*
paper/*
```

Commit code, configs, docs, and scripts. Do not commit datasets, pretrained
weights, raw experiment outputs, generated CSV summaries, or visualization
figures unless explicitly needed for a release or report artifact. The paper is
also ignored in this working copy because it is managed as a local/Overleaf
artifact rather than as part of the code repository.
