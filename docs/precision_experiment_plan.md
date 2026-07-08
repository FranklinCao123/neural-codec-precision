# Precision Experiment Plan

This project studies deployment-oriented numerical precision sensitivity in
learned image compression. The main question is not which codec has the best
FP32 rate-distortion performance, but how different architectural components
behave under reduced precision.

The target comparison is:

- convolution-dominated transform modules;
- autoregressive and context entropy modules;
- attention-enhanced modules;
- modern efficient context models;
- Transformer-CNN mixed models.

The evaluation must use the full codec path whenever possible:

```text
compress -> bitstream -> decompress -> reconstruction
```

Forward-only timing or deployment tests are secondary and should not replace
full codec RD evaluation.

## Model Coverage

The current model set should cover the following roles.

| Role | Model |
| --- | --- |
| Classic CNN + hyperprior | BMShj2018-hyperprior |
| Classic autoregressive context | Minnen2018 / MBT2018 |
| Cheng context/GMM control | Cheng2020-anchor |
| Attention + GMM/context | Cheng2020-attn |
| Modern efficient context codec | ELIC |
| Transformer-CNN mixed codec | TCM / LIC-TCM |

Minnen2018 is available through the CompressAI `mbt2018` implementation and can
be run with the `minnen2018` config alias. ELIC and TCM require external model
adapters because they are not part of the current CompressAI zoo loader.

## Core Experiments

Run these settings for every new model.

```text
FP32 baseline
FP16 inference
BF16 inference
INT8 weight-only PTQ
INT8 weight+activation calibrated PTQ
INT12 fixed-point activation
INT10 fixed-point activation
```

These experiments answer whether reduced precision changes bpp, PSNR, MS-SSIM,
runtime, memory, or numerical validity.

If INT10 fixed-point produces invalid reconstructions or a large quality/rate
drop, add boundary tests:

```text
INT16 fixed-point activation
INT8 fixed-point activation
```

## Module Ablations

Run calibrated INT8 weight+activation PTQ module ablations for:

```text
g_a
g_s
h_a + h_s
all neural transforms
```

When the model implementation exposes finer modules, add:

```text
context / entropy-parameter network
attention / Transformer blocks
CNN branch and Transformer branch for TCM
grouped context modules for ELIC
```

The purpose is to separate rate-sensitive and distortion-sensitive modules.
The expected interpretation is:

```text
g_a / context / entropy path -> bpp sensitivity
g_s -> reconstruction quality sensitivity
attention / Transformer / context -> possible extra low-precision sensitivity
```

## Error Analysis

For each added model, run activation/module precision analysis and collect:

```text
relative MSE
SQNR
saturation ratio
worst INT8 layers
```

Summaries should be grouped by top-level path and module type, for example:

```text
Conv
GDN
Residual
MaskedConv / context
entropy-parameter network
Attention
Transformer block
normalization / MLP
```

These metrics explain why a precision setting changes bpp or reconstruction
quality.

## Immediate Execution Order

1. Run Minnen2018 core experiments from the new configs.
2. Run Minnen2018 module ablations.
3. Add an ELIC adapter and run the same core experiments.
4. Add a TCM adapter and run the same core experiments.
5. Add ELIC/TCM module ablations for context and attention/Transformer modules.
6. Rebuild result tables and figures.
7. Update the paper narrative and conclusions.

