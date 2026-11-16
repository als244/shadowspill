"""Apply scheduled values to an ordinary optimizer or named model buffer."""

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


def set_hyperparams(
    model: nn.Module, optimizer: torch.optim.Optimizer, values: Mapping[str, Any]
) -> None:
    buffers = dict(model.named_buffers())
    for name, value in values.items():
        key = name
        if name.startswith("optimizer_group."):
            parts = name.split(".", 2)
            if len(parts) != 3:
                raise KeyError(f"expected optimizer_group.<index>.<key>: {name!r}")
            _, index, key = parts
            if not index.isdecimal() or int(index) >= len(optimizer.param_groups):
                raise KeyError(f"no optimizer group named {name!r}")
            selected = optimizer.param_groups[int(index)]
            groups = [selected] if key in selected else []
        else:
            groups = [group for group in optimizer.param_groups if key in group]
        buffer = buffers.get(name)
        if groups and buffer is not None:
            raise ValueError(
                f"{name!r} names both an optimizer value and a model buffer"
            )
        if buffer is not None:
            with torch.no_grad():
                buffer.fill_(value)
        elif groups:
            for group in groups:
                held = group[key]
                if isinstance(held, torch.Tensor):
                    held.fill_(value)
                elif isinstance(held, (tuple, list)):
                    items = (
                        [value] * len(held)
                        if isinstance(value, (int, float))
                        else list(value)
                    )
                    if len(items) != len(held):
                        raise ValueError(f"{name!r} needs {len(held)} values per group")
                    group[key] = type(held)(items)
                else:
                    group[key] = value
        else:
            raise KeyError(f"no optimizer value or model buffer named {name!r}")
