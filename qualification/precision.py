"""Shared dtype flags and reporting for numerical and performance gates."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

from workloads.precision import DTYPE_FIELDS, FLOAT_DTYPES, TrainingDtypes


def add_dtype_arguments(parser: argparse.ArgumentParser) -> None:
    help_text = {
        "model_dtype": "model weights (default: BF16; FP16 below SM80)",
        "master_dtype": "master weights (default: none)",
        "grad_dtype": "accumulated gradients (default: model weight dtype)",
        "opt_state_dtype": "optimizer moments (default: BF16; FP32 below SM80)",
    }
    for name in DTYPE_FIELDS:
        choices = FLOAT_DTYPES
        if name == "master_dtype":
            choices = ("none", *choices)
        elif name == "grad_dtype":
            choices = ("parameter", *choices)
        parser.add_argument(
            "--" + name.replace("_", "-"),
            choices=choices,
            help=help_text[name],
        )


def detect_device_precision(*, allow_cpu: bool = False) -> dict[str, object]:
    """Read the selected device without initializing this process's allocator.

    CUDA_VISIBLE_DEVICES and the environment are inherited. Only the probe
    process opens a device context; it exits before this worker creates its
    runtime. Matrix launchers resolve once and forward all four dtype choices.
    allow_cpu keeps the suite usable without an accelerator; CPU-only fixtures
    retain their existing BF16 storage default.
    """
    source = (
        "import json, torch; "
        + (
            "import sys; "
            "torch.cuda.is_available() or "
            "(print(json.dumps({'device': 'cpu', 'bf16': True})), sys.exit(0)); "
            if allow_cpu
            else ""
        )
        + "device = torch.cuda.current_device(); "
        "bf16 = (torch.cuda.get_device_capability(device)[0] >= 8 "
        "if torch.version.cuda else "
        "torch.cuda.is_bf16_supported(including_emulation=False)); "
        "print(json.dumps({'device': torch.cuda.get_device_name(device), "
        "'bf16': bf16}))"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", source],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "cannot select gate defaults for the visible accelerator:\n"
            + error.stderr.strip()
        ) from error
    return json.loads(result.stdout)


def resolve_dtype_defaults(
    options: argparse.Namespace,
    *,
    hardware: dict[str, object] | None = None,
) -> None:
    """Fill unspecified flags from hardware, preserving explicit overrides."""
    if getattr(options, "case_factory", None) is not None:
        return
    if options.model_dtype is None or options.opt_state_dtype is None:
        hardware = detect_device_precision() if hardware is None else hardware
        supports_bf16 = bool(hardware["bf16"])
        if options.model_dtype is None:
            options.model_dtype = "bfloat16" if supports_bf16 else "float16"
        if options.opt_state_dtype is None:
            options.opt_state_dtype = "bfloat16" if supports_bf16 else "float32"
    if options.master_dtype is None:
        options.master_dtype = "none"
    if options.grad_dtype is None:
        options.grad_dtype = options.model_dtype


def dtype_overrides(options: Any) -> dict[str, str]:
    return {
        name: str(value)
        for name in DTYPE_FIELDS
        if (value := getattr(options, name, None)) is not None
    }


def dtype_arguments(options: Any) -> list[str]:
    return [
        item
        for name, value in dtype_overrides(options).items()
        for item in ("--" + name.replace("_", "-"), value)
    ]


def dtype_description(options: Any) -> str:
    overrides = dtype_overrides(options)
    if getattr(options, "case_factory", None) is not None:
        return "DTYPES: custom factory (case_options); " + (
            ", ".join(f"{name}={value}" for name, value in overrides.items())
            or "no gate overrides"
        )
    return "DTYPES: " + TrainingDtypes(**overrides).description()
