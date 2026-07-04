# LIC Precision Experiments

This project is a scaffold for precision experiments on learned image compression models.

Initial target:

- Model: Cheng2020 / CompressAI pretrained model
- Dataset: Kodak first, CLIC later
- Precisions: FP32 baseline, FP16, INT8 PTQ, INT8 QAT, mixed precision
- Metrics: bpp, PSNR, MS-SSIM, BD-rate, encode/decode time, model size

## Workflow

1. Run an FP32 baseline.
2. Add FP16 inference.
3. Add INT8 post-training quantization.
4. Add QAT fine-tuning if PTQ loses too much RD performance.
5. Run module-wise quantization analysis.
6. Generate tables and rate-distortion curves.

## Layout

```text
configs/        Experiment configs.
data/           Local datasets, ignored by git.
models/         Model loading and wrappers.
quantization/   FP16, PTQ, QAT, fixed-point utilities.
evaluation/     Metrics, codec evaluation, BD-rate, benchmark code.
scripts/        Command-line entry points.
results/        Generated tables and figures, ignored by git except placeholders.
docs/           Notes and experiment plans.
```

## Next Step

Install PyTorch, CompressAI, and metric dependencies, then implement the FP32 Cheng2020 baseline script.
