"""Translate the shared baseline precision settings into build/optimizer options."""

import torch


def optimizer_arguments(precision):
    arguments = {
        "gradient_dtype": getattr(torch, precision.grad_dtype or precision.model_dtype),
        "parameter_rounding": precision.parameter_rounding or "nearest",
        "opt_state_rounding": precision.opt_state_rounding or "nearest",
    }
    if precision.opt_state_dtype is not None:
        arguments["opt_state_dtype"] = (
            "parameter"
            if precision.opt_state_dtype == "parameter"
            else getattr(torch, precision.opt_state_dtype)
        )
    return arguments


def build_arguments(precision):
    return {
        "grad_dtype": getattr(torch, precision.grad_dtype or precision.model_dtype),
        "master_dtype": None
        if precision.master_dtype is None
        else getattr(torch, precision.master_dtype),
        "fuse_narrow_gradient_accumulation": (
            precision.fuse_narrow_gradient_accumulation
        ),
    }
