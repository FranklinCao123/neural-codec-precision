# External Model Integration

ELIC and TCM are not loaded from the current CompressAI model zoo helper. To
evaluate them with the same scripts, clone the external implementation on the
server, make sure its model class exposes the standard codec methods, and use
`source: external_python` in the experiment config.

The model object should support the APIs used by `evaluation/eval_codec.py`:

```text
model(x)                         # forward timing
model.compress(x)                # bitstream generation
model.decompress(strings, shape) # reconstruction
model.update(force=True)         # optional entropy model table update
```

If an implementation does not expose compatible `compress` and `decompress`
methods, it can still be used for forward-only deployment analysis, but it
should not be mixed into the full-codec RD tables.

## Config Template

```yaml
experiment:
  name: elic_fp32_baseline
  seed: 1234

model:
  source: external_python
  name: elic
  python_path: external/ELIC
  module: path.to.model_module
  class: ModelClassOrFactory
  init_args:
    quality: 3
  checkpoint: checkpoints/elic_quality3.pth
  state_dict_key: auto
  strict_state_dict: true
  call_update: true

data:
  dataset: kodak
  root: data/kodak
  batch_size: 1
  num_workers: 0

precision:
  mode: fp32

evaluation:
  metrics:
    - bpp
    - psnr
    - ms_ssim
  benchmark_forward: true
  benchmark_memory: true
  forward_warmup: 1
  forward_repeats: 3
  save_reconstructions: false

output:
  dir: results/raw/elic_fp32
```

Fields to edit per external repo:

```text
python_path      repository root to add to PYTHONPATH
module           Python module containing the model class or builder
class/function   model class or factory function
init_args        constructor arguments required by that repo
checkpoint       pretrained checkpoint path
state_dict_key   checkpoint key; use auto first, then set explicitly if needed
```

## Recommended Order

First validate the FP32 full codec path:

```bash
python scripts/run_fp32_baseline.py --config configs/elic_fp32.yaml --device cuda
```

Accept the model only if:

```text
avg_psnr is finite
avg_ms_ssim is finite
num_invalid_reconstructions = 0
```

After the FP32 baseline is valid, add the same core precision configs:

```text
FP16
BF16
INT8 weight-only PTQ
INT8 weight+activation calibrated PTQ
INT12 fixed-point activation
INT10 fixed-point activation
```

Then add module ablations only for modules that can be selected cleanly by
top-level name, such as:

```text
g_a
g_s
h_a + h_s
context / entropy-parameter path
attention / Transformer blocks
```

