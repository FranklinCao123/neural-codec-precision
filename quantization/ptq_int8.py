"""Post-training INT8 quantization helpers."""


def prepare_ptq(model, config: dict):
    """Prepare a model for calibration."""
    raise NotImplementedError("INT8 PTQ preparation is not implemented yet.")


def convert_ptq(model):
    """Convert a calibrated model to INT8."""
    raise NotImplementedError("INT8 PTQ conversion is not implemented yet.")
