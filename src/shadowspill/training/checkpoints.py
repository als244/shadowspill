"""Model checkpoint helpers for eager reference training.

A checkpoint contains ``model``, ``step`` and, by default, ``optimizer``: the
model's state dict, completed updates and optional optimizer state. A weight
trained over a master copy at another precision is written under the weight's
name by default. With weights="compute", only the compute representation is
saved and restored masters take its upcast values. Either choice stores weights
once and loads into a plain model of either precision
as it stands. Saved Executor checkpoints instead use the bounded archive
interface in :mod:`shadowspill.checkpoint`.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn

from shadowspill.pytorch.state.serialization import (
    decode_tensor_state,
    encode_tensor_state,
)


def save(
    path: Path,
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    masters: Mapping[str, torch.Tensor] | None = None,
    *,
    weights: Literal["master", "compute"] = "master",
    include_optimizer: bool = True,
    frozen_state_id: str | None = None,
    mutable_frozen: Collection[str] = (),
) -> None:
    """Write one weight representation and the optimizer's state.

    ``master`` uses a master where available; ``compute`` uses model weights.
    Loading casts the saved values into both destinations when masters exist.
    With ``frozen_state_id``, omit frozen parameters except ``mutable_frozen``;
    retain every persistent buffer. The execution backend identifies mutations.
    ``include_optimizer=False`` omits optimizer state independently of precision.
    """
    if weights not in {"master", "compute"}:
        raise ValueError("weights must be 'master' or 'compute'")
    validate_frozen_state_id(frozen_state_id)
    model = module.state_dict()
    if frozen_state_id is not None:
        omitted = frozen_parameter_names(module) - set(mutable_frozen)
        model = {name: value for name, value in model.items() if name not in omitted}
    for name, master in (masters or {}).items() if weights == "master" else ():
        for alias in _names(module)[name]:
            model[alias] = master.detach()
    torch.save(
        {
            "model": encode_tensor_state(model),
            **({"optimizer": optimizer.state_dict()} if include_optimizer else {}),
            "step": step,
            **({} if frozen_state_id is None else {"frozen_state_id": frozen_state_id}),
        },
        path,
    )


def load(
    state: dict[str, Any],
    module: nn.Module,
    optimizer: torch.optim.Optimizer,
    masters: Mapping[str, torch.Tensor] | None = None,
    *,
    frozen_state_id: str | None = None,
) -> int:
    """Restore a checkpoint into the module, its masters and the optimizer;
    return its step. A master takes the value written for its weight, and the
    weight that value's cast. Missing optimizer state resets the optimizer;
    the completed-update count is still restored."""

    omitted = omitted_frozen_parameters(state, module, frozen_state_id)
    model = decode_tensor_state(state["model"], module.state_dict())
    dtypes = {name: value.dtype for name, value in module.state_dict().items()}
    with torch.no_grad():
        for name, master in (masters or {}).items():
            master.copy_(model[name])
            for alias in _names(module)[name]:
                model[alias] = model[alias].to(dtypes[alias])
    missing = module.load_state_dict(model, strict=not omitted)
    if set(missing.missing_keys) != omitted or missing.unexpected_keys:
        raise ValueError("checkpoint state inventory differs")
    if "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    else:
        optimizer.state.clear()
    return int(state["step"])


def validate_frozen_state_id(identity: str | None) -> None:
    if identity is not None and (not isinstance(identity, str) or not identity.strip()):
        raise ValueError("frozen_state_id must be a nonempty string")


def frozen_parameter_names(module: nn.Module) -> set[str]:
    return {
        name
        for name, parameter in module.named_parameters(remove_duplicate=False)
        if not parameter.requires_grad
    }


def omitted_frozen_parameters(
    state: Mapping[str, Any],
    module: nn.Module,
    frozen_state_id: str | None,
    *,
    missing_parameters: Collection[str] = (),
) -> set[str]:
    """Validate a partial restore; only unchanged frozen parameters may be absent."""
    validate_frozen_state_id(frozen_state_id)
    if state.get("frozen_state_id") != frozen_state_id:
        raise ValueError(
            "checkpoint frozen_state_id differs; initialize the same frozen base"
        )
    if frozen_state_id is None:
        return set()
    missing = set(module.state_dict()) - set(state["model"])
    frozen = frozen_parameter_names(module)
    if missing - frozen - set(missing_parameters) or set(state["model"]) - set(
        module.state_dict()
    ):
        raise ValueError(
            "checkpoint must retain every trainable parameter and persistent buffer"
        )
    return missing & frozen


def _names(module: nn.Module) -> dict[str, tuple[str, ...]]:
    """Every name a weight goes by, under the first of them."""

    names: dict[int, list[str]] = {}
    for name, parameter in module.named_parameters(remove_duplicate=False):
        names.setdefault(id(parameter), []).append(name)
    return {every[0]: tuple(every) for every in names.values()}
