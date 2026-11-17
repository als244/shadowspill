"""Optimizer defaults shared by text benchmarks, training and eager references.

The benchmark/reference policy prefers stochastic rounding for BF16 MLOps AdamW
moments. This is an explicit workload policy; the generic training API does not
inspect optimizer implementation names or change their numerical defaults.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from functools import partial
from itertools import groupby
from typing import Any

import torch


def build_optimizer(
    constructor: Callable[..., torch.optim.Optimizer],
    parameters: Iterable[Any],
    arguments: Mapping[str, Any],
) -> torch.optim.Optimizer:
    """Apply the benchmark/reference optimizer defaults, preserving explicit choices.

    Explicit constructor and group rounding choices win. Other optimizers and
    caller-owned optimizer instances retain their own semantics. Mixed moment
    dtypes under ``opt_state_dtype="parameter"`` get consecutive dtype groups
    without changing parameter order or stochastic rounding salts.
    """
    target = constructor
    supplied = dict(arguments)
    while isinstance(target, partial):
        supplied = {**target.keywords, **supplied}
        target = target.func
    if (
        getattr(target, "implementation_id", None) != "builtin.adamw.triton"
        or "opt_state_rounding" in supplied
    ):
        return constructor(parameters, **arguments)

    values = list(parameters)
    if not values:
        return constructor(values, **arguments)
    groups = values if values and isinstance(values[0], dict) else [{"params": values}]
    configured = []
    for original in groups:
        group = dict(original)
        group["params"] = list(group["params"])
        dtype = group.get(
            "opt_state_dtype", supplied.get("opt_state_dtype", torch.bfloat16)
        )
        if "opt_state_rounding" not in group:
            if dtype == torch.bfloat16:
                group["opt_state_rounding"] = "stochastic"
            elif dtype == "parameter" and group["params"]:
                for bf16, entries in groupby(
                    enumerate(group["params"]),
                    key=lambda entry: entry[1].dtype == torch.bfloat16,
                ):
                    indices, members = zip(*entries, strict=True)
                    part = {
                        **group,
                        "params": list(members),
                        "opt_state_rounding": "stochastic" if bf16 else "nearest",
                    }
                    if "param_names" in group:
                        part["param_names"] = [group["param_names"][i] for i in indices]
                    configured.append(part)
                continue
        configured.append(group)
    return constructor(configured, **arguments)
