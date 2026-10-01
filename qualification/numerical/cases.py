"""Numerical qualification composition: user factories or supplied model recipes."""

from __future__ import annotations

import importlib
from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_DEVICE_BUDGETS = {"llama3": 10 << 30, "qwen35": 10 << 30, "olmoe": 8 << 30}


def build_case(
    family: str,
    *,
    model_implementation: str = "pytorch",
    seed: int = 20_260_811,
    model_config: Mapping[str, Any] | None = None,
    data_geometry: Sequence[Mapping[str, Any]] | None = None,
    case_factory: str | None = None,
    case_options: Mapping[str, Any] | None = None,
    model_dtype: str | None = None,
    master_dtype: str | None = None,
    grad_dtype: str | None = None,
    opt_state_dtype: str | None = None,
) -> Any:
    """Keep catalog imports out of the user-supplied factory path."""
    if case_factory is not None and (
        model_dtype is not None or opt_state_dtype is not None
    ):
        raise ValueError("custom factories configure dtypes through case_options")
    if case_factory is not None:
        module_name, separator, attribute = case_factory.partition(":")
        if separator == "" or not module_name or not attribute:
            raise ValueError("case_factory must use the form 'module:function'")
        factory = getattr(importlib.import_module(module_name), attribute)
        if not callable(factory):
            raise TypeError(
                f"qualification case factory is not callable: {case_factory}"
            )
        case = factory(
            model_name=family,
            model_implementation=model_implementation,
            seed=seed,
            model_config=dict(model_config or {}),
            data_geometry=tuple(dict(item) for item in (data_geometry or ())),
            case_options=dict(case_options or {}),
        )
        required = (
            "family",
            "model_implementation",
            "model",
            "microbatches",
            "objective",
            "optimizer",
            "implementations",
        )
        missing = [name for name in required if not hasattr(case, name)]
        if missing:
            raise TypeError(
                f"qualification case factory {case_factory} omitted: "
                + ", ".join(missing)
            )
        return case
    if case_options:
        raise ValueError("case_options require a custom case_factory")
    from workloads.numerical import build_case as supplied_case

    return supplied_case(
        family,
        model_implementation=model_implementation,
        seed=seed,
        model_config=model_config,
        data_geometry=data_geometry,
        model_dtype=model_dtype,
        master_dtype=master_dtype,
        grad_dtype=grad_dtype,
        opt_state_dtype=opt_state_dtype,
    )
