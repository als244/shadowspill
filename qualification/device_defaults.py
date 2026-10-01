"""Small-device gate defaults, resolved before either worker creates tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .precision import detect_device_precision, resolve_dtype_defaults


def _nonnegative_mib(value: str) -> int:
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("external headroom must be nonnegative")
    return result


def add_memory_budget_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--external-headroom-mib",
        type=_nonnegative_mib,
        help=(
            "external memory reserved inside the execution cap "
            "(default: 512; 0 reserves no allowance)"
        ),
    )
    parser.add_argument(
        "--reject-overbudget",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "reject external and whole-process memory overruns "
            "(default: disabled; pool bounds remain enforced)"
        ),
    )


def _memory_budget_defaults(options: argparse.Namespace) -> None:
    if not isinstance(options.reject_overbudget, bool):
        raise ValueError("reject-overbudget must be a bool")
    if getattr(options, "external_headroom_mib", None) is None:
        options.external_headroom_mib = 512
    if options.external_headroom_mib < 0:
        raise ValueError("external-headroom-mib must be nonnegative")


# Below SM80, smaller models use a smaller execution pool.
SMALL_NUMERICAL_BUDGETS = {
    "llama3": 3 << 30,
    "qwen35": 3 << 30,
    "olmoe": 3 << 30,
}


def numerical_defaults(
    options: argparse.Namespace,
    *,
    hardware: dict[str, object] | None = None,
) -> dict[str, int]:
    """Return budget defaults and merge small presets below SM80 only.

    Explicit model fields, budgets, precision and reference directories win.
    The ordinary preset objects and SM80+ reference identities stay unchanged.
    Custom factories own their geometry and precision choices.
    """
    from qualification.numerical.cases import DEFAULT_DEVICE_BUDGETS

    from .numerical.references import DEFAULT_REFERENCE_DIRECTORY

    if options.case_factory is not None:
        return dict(DEFAULT_DEVICE_BUDGETS)
    hardware = detect_device_precision() if hardware is None else hardware
    resolve_dtype_defaults(options, hardware=hardware)
    _memory_budget_defaults(options)
    if bool(hardware["bf16"]):
        return dict(DEFAULT_DEVICE_BUDGETS)
    text = options.model_config
    source = Path(text[1:]).expanduser().read_text() if text.startswith("@") else text
    values = json.loads(source)
    if not isinstance(values, dict):
        raise ValueError("model config must decode to an object")
    options.model_config = json.dumps({"n_layers": 4, "vocab_size": 8192, **values})
    if options.reference_dir == DEFAULT_REFERENCE_DIRECTORY:
        options.reference_dir = Path("qualification/results/references/pre_sm80")
    return dict(SMALL_NUMERICAL_BUDGETS)


def performance_defaults(
    options: argparse.Namespace,
    *,
    hardware: dict[str, object] | None = None,
) -> None:
    """Keep 16 GiB on SM80+; default to 10 GiB below SM80."""
    hardware = detect_device_precision() if hardware is None else hardware
    resolve_dtype_defaults(options, hardware=hardware)
    _memory_budget_defaults(options)
    if options.execution_budget_gib is None:
        options.execution_budget_gib = 16 if bool(hardware["bf16"]) else 10
