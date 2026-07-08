"""CompressAI model loading helpers."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch


def _require_compressai():
    try:
        import compressai.zoo as zoo
    except ImportError as exc:
        raise ImportError(
            "CompressAI is required to load pretrained learned image compression "
            "models. Install it on the server before running experiments."
        ) from exc
    return zoo


def _normalize_model_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def load_model_from_config(config: dict):
    """Load a codec model described by an experiment config."""
    model_cfg = config.get("model", {})
    source = model_cfg.get("source", "compressai")
    if source == "external_python":
        return _load_external_python_model(model_cfg)
    if source != "compressai":
        raise ValueError(
            f"Unsupported model source: {source!r}. "
            "Supported sources: compressai, external_python"
        )

    zoo = _require_compressai()
    name = _normalize_model_name(model_cfg.get("name", "cheng2020-anchor"))
    quality = int(model_cfg.get("quality", 3))
    metric = model_cfg.get("metric", "mse")
    pretrained = bool(model_cfg.get("pretrained", True))

    constructor_names = {
        "cheng2020-anchor": "cheng2020_anchor",
        "cheng2020-attn": "cheng2020_attn",
        "bmshj2018-factorized": "bmshj2018_factorized",
        "bmshj2018-hyperprior": "bmshj2018_hyperprior",
        "minnen2018": "mbt2018",
        "minnen2018-mean": "mbt2018_mean",
        "mbt2018-mean": "mbt2018_mean",
        "mbt2018": "mbt2018",
    }
    constructors = {
        model_name: getattr(zoo, attr_name)
        for model_name, attr_name in constructor_names.items()
        if hasattr(zoo, attr_name)
    }
    if name not in constructors:
        if name in {"elic", "tcm", "lic-tcm"}:
            raise ValueError(
                f"Model {name!r} is not available through the current CompressAI "
                "zoo loader. Add an external model adapter before running this "
                "experiment."
            )
        supported = ", ".join(sorted(constructors))
        raise ValueError(f"Unsupported CompressAI model {name!r}. Supported: {supported}")

    model = constructors[name](
        quality=quality,
        metric=metric,
        pretrained=pretrained,
    )

    # Required before real entropy coding through compress/decompress.
    if hasattr(model, "update"):
        model.update(force=True)

    return model


def _load_external_python_model(model_cfg: dict):
    """Load a model from an external Python module.

    The external object must return a torch.nn.Module-compatible codec with
    compress/decompress methods if it is used for full codec evaluation.
    """
    python_path = model_cfg.get("python_path")
    if python_path:
        resolved = Path(python_path).expanduser().resolve()
        if str(resolved) not in sys.path:
            sys.path.insert(0, str(resolved))

    module_name = model_cfg.get("module")
    object_name = model_cfg.get("class") or model_cfg.get("function")
    if not module_name or not object_name:
        raise ValueError(
            "external_python models require `model.module` and `model.class` "
            "or `model.function`."
        )

    module = importlib.import_module(module_name)
    factory = getattr(module, object_name)
    init_args = model_cfg.get("init_args", {}) or {}
    model = factory(**init_args)

    checkpoint_path = model_cfg.get("checkpoint")
    if checkpoint_path:
        checkpoint = torch.load(
            Path(checkpoint_path).expanduser(),
            map_location=model_cfg.get("checkpoint_map_location", "cpu"),
        )
        state_dict = _extract_state_dict(
            checkpoint,
            key=model_cfg.get("state_dict_key", "auto"),
        )
        strip_prefix = model_cfg.get("strip_state_dict_prefix")
        if strip_prefix:
            state_dict = {
                key[len(strip_prefix) :] if key.startswith(strip_prefix) else key: value
                for key, value in state_dict.items()
            }
        missing, unexpected = model.load_state_dict(
            state_dict,
            strict=bool(model_cfg.get("strict_state_dict", True)),
        )
        if missing or unexpected:
            print(
                "Loaded external checkpoint with state-dict differences: "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )

    if bool(model_cfg.get("call_update", True)) and hasattr(model, "update"):
        model.update(force=True)

    return model


def _extract_state_dict(checkpoint, key: str):
    if key and key != "auto":
        return checkpoint[key]
    if isinstance(checkpoint, dict):
        for candidate in ("state_dict", "model", "net", "network", "model_state_dict"):
            value = checkpoint.get(candidate)
            if isinstance(value, dict):
                return value
    return checkpoint
