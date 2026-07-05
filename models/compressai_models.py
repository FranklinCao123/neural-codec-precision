"""CompressAI model loading helpers."""

from __future__ import annotations


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
    """Load a CompressAI model described by an experiment config."""
    model_cfg = config.get("model", {})
    source = model_cfg.get("source", "compressai")
    if source != "compressai":
        raise ValueError(f"Unsupported model source: {source!r}")

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
        "mbt2018-mean": "mbt2018_mean",
        "mbt2018": "mbt2018",
    }
    constructors = {
        model_name: getattr(zoo, attr_name)
        for model_name, attr_name in constructor_names.items()
        if hasattr(zoo, attr_name)
    }
    if name not in constructors:
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
