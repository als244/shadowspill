"""Caller-owned numerical settings in compiled identity and phase admission.

Reading these settings never initializes CUDA. They describe process configuration,
not a promise of bitwise invariance across matrix shapes or kernel providers.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterable, Sequence
from functools import lru_cache
from typing import TYPE_CHECKING, Any, cast

import torch

if TYPE_CHECKING:
    from shadowspill.artifacts.program import Program

_PREFIX = "torch.numerical_settings."
_VERSION = _PREFIX + "v1:"
_ENVIRONMENT = (
    "CUBLAS_WORKSPACE_CONFIG",
    "CUBLASLT_WORKSPACE_SIZE",
    "NVIDIA_TF32_OVERRIDE",
    "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE",
)


def _settings() -> dict[str, bool | int | str | None]:
    matmul = torch.backends.cuda.matmul
    # Torch's annotations omit the runtime conv/rnn precision properties.
    cudnn: Any = torch.backends.cudnn
    precision = matmul.fp32_precision
    # Legacy high and medium both resolve to the newer TF32 setting. Retain
    # their distinction when readable. Setting only the new precision API
    # can make the legacy getter refuse mixed APIs, even without a legacy write.
    legacy_precision: str | None
    try:
        legacy_precision = torch.get_float32_matmul_precision()
    except RuntimeError as error:
        if "mix of the legacy and new APIs" not in str(error):
            raise
        legacy_precision = None
    return {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "float32_matmul_precision": legacy_precision,
        # An unset new flag with the default legacy highest precision is IEEE.
        # Resetting the legacy flag explicitly must not make a cold default
        # worker look incompatible with that same effective configuration.
        "cuda.matmul.fp32_precision": "ieee" if precision == "none" else precision,
        **{
            f"cuda.matmul.{name}": getattr(matmul, name)
            for name in (
                "allow_bf16_reduced_precision_reduction",
                "allow_bf16_reduced_precision_reduction_split_k",
                "allow_fp16_reduced_precision_reduction",
                "allow_fp16_reduced_precision_reduction_split_k",
                "allow_fp16_accumulation",
            )
        },
        "cuda.preferred_blas_library": str(
            torch.backends.cuda.preferred_blas_library()
        ),
        "cudnn.enabled": cudnn.enabled,
        "cudnn.deterministic": cudnn.deterministic,
        "cudnn.benchmark": cudnn.benchmark,
        "cudnn.benchmark_limit": cudnn.benchmark_limit,
        "cudnn.conv.fp32_precision": cudnn.conv.fp32_precision,
        "cudnn.rnn.fp32_precision": cudnn.rnn.fp32_precision,
        **{f"env.{name}": os.environ.get(name) for name in _ENVIRONMENT},
    }


def capture_numerical_settings() -> str:
    return _VERSION + json.dumps(_settings(), sort_keys=True, separators=(",", ":"))


@lru_cache(maxsize=128)
def _decode(contract: str) -> dict[str, bool | int | str | None]:
    if not contract.startswith(_VERSION):
        raise ValueError(
            "unsupported Torch numerical settings version; rebuild Program"
        )
    try:
        settings = json.loads(contract[len(_VERSION) :])
    except ValueError as error:
        raise ValueError("invalid Torch numerical settings; rebuild Program") from error
    current = _settings()
    nullable = {"float32_matmul_precision", *(f"env.{name}" for name in _ENVIRONMENT)}
    if (
        not isinstance(settings, dict)
        or settings.keys() != current.keys()
        or any(
            type(value) not in (str, type(None))
            if name in nullable
            else type(value) is not type(current[name])
            for name, value in settings.items()
        )
    ):
        raise ValueError("invalid Torch numerical settings schema; rebuild Program")
    return cast(dict[str, bool | int | str | None], settings)


def _validate(contracts: Iterable[str]) -> None:
    snapshots = {item for item in contracts if item.startswith(_PREFIX)}
    if not snapshots:
        return  # Legacy artifacts have no claim about their process configuration.
    decoded = [_decode(item) for item in sorted(snapshots)]
    if any(item != decoded[0] for item in decoded[1:]):
        raise ValueError("tasks require conflicting Torch numerical settings")
    actual = _settings()
    differences = [
        f"{name}: expected {expected!r}, actual {actual[name]!r}"
        for name, expected in sorted(decoded[0].items())
        if expected != actual[name]
    ]
    if differences:
        raise RuntimeError(
            "Torch numerical settings differ from compiled tasks: "
            + "; ".join(differences)
            + ". Restore the build settings, or rebuild, remeasure and replan. "
            "Configure environment variables before starting the worker."
        )


def validate_numerical_settings(
    contracts: Iterable[str], *, check: Callable[[str | None], None] | None = None
) -> None:
    """Refuse changed arithmetic before launching tasks; never change settings.

    Build/profiling resource groups can coordinate the refusal before any rank
    enters provider collectives. Execution coordinates its entire prepare step.
    """
    if check is None:
        _validate(contracts)
        return
    failure: Exception | None = None
    try:
        _validate(contracts)
    except Exception as error:
        failure = error
    try:
        check(None if failure is None else str(failure))
    except Exception as error:
        raise error from failure
    if failure is not None:
        raise failure


def compose_numerical_contracts(children: Iterable[Sequence[str]]) -> tuple[str, ...]:
    children = tuple(children)
    configured = [any(item.startswith(_PREFIX) for item in child) for child in children]
    if any(configured) and not all(configured):
        raise ValueError("cannot compose tasks with missing Torch numerical settings")
    contracts = tuple(sorted({item for child in children for item in child}))
    validate_numerical_settings(contracts)
    return contracts


def program_numerical_contracts(
    program: Program, keys: Iterable[str] | None = None
) -> tuple[str, ...]:
    metadata = program.bindings.get("task_metadata", {})
    keys = program.executables if keys is None else keys
    return tuple(
        sorted(
            {
                item
                for key in keys
                for item in metadata.get(key, {}).get("numerical_contracts", ())
            }
        )
    )
