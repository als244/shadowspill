"""Optimizer arguments a schedule drives become tensors the compiled step reads."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import torch

GROUP = re.compile(r"^optimizer_group\.0\.(\w+)$")


def scheduled_arguments(
    arguments: Mapping[str, Any], schedules: Mapping[str, Any]
) -> dict[str, Any]:
    """``arguments`` with every number a schedule names made a scalar tensor:
    a Python number is fixed into the compiled step at capture, a tensor stays
    writable, so a schedule can only move the latter. The runner builds one
    parameter group, ``optimizer_group.0``; a hyperparameter no schedule names
    stays the constant it is given as."""

    result = dict(arguments)
    for name in schedules:
        match = GROUP.match(name)
        if match is None:
            continue
        key = match.group(1)
        value = result.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            result[key] = torch.tensor(float(value))
    return result
