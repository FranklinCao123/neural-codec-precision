# Experiment Plan

## Goal

Measure how numerical precision affects learned image compression models.

## Main Questions

- How much RD performance is lost by FP16, INT8 PTQ, and INT8 QAT?
- Is Cheng2020 robust to low-precision inference?
- Which modules are most precision-sensitive: encoder, decoder, hyperprior, entropy model, or attention?
- Does QAT recover the degradation introduced by INT8 PTQ?

## Minimal First Milestone

Run Cheng2020 FP32 on Kodak and produce:

- bpp
- PSNR
- MS-SSIM
- encode time
- decode time

## Later Milestones

1. Add FP16 inference.
2. Add INT8 PTQ with calibration.
3. Add QAT fine-tuning.
4. Add module-wise mixed precision.
5. Add a second model such as ELIC or LIC-TCM.
