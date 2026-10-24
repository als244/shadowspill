"""Validate the complete call structure and stage rank-local examples in spill."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import torch
from torch.utils._pytree import tree_flatten, treespec_dumps

from shadowspill.artifacts._values import freeze
from shadowspill.pytorch.guards import capture_input_signature

from .storage import Storage


def stage_inputs(
    state: Storage,
    args: Sequence[tuple[Any, ...]] | None,
    kwargs: Sequence[Mapping[str, Any]] | None,
    input_loader: Callable[[int], tuple[tuple[Any, ...], Mapping[str, Any]]] | None,
) -> None:
    bindings = state.plan.program.bindings
    count = bindings.get("num_examples", 1)
    if input_loader is not None:
        if args is not None or kwargs is not None:
            raise ValueError("supply args/kwargs or input_loader, not both")
    elif (
        args is None
        or len(args) != count
        or (kwargs is not None and len(kwargs) != count)
    ):
        raise ValueError(f"run requires exactly {count} examples")
    slots_by_example = bindings["root_inputs"]
    if bindings["kind"] == "forward":
        slots_by_example = [slots_by_example]
    for index, slots in enumerate(slots_by_example):
        if input_loader is not None:
            values = input_loader(index)
        else:
            assert args is not None
            values = (args[index], {} if kwargs is None else kwargs[index])
        signature = capture_input_signature(values)
        identity = {
            "tree": treespec_dumps(signature.tree_spec),
            "leaves": [leaf.identity() for leaf in signature.leaves],
        }
        if freeze(identity) != bindings["input_signature"]:
            raise ValueError(f"example {index} differs from the built input signature")
        leaves, _ = tree_flatten(values)
        copied = set()
        for slot in slots:
            source = bindings["root_sources"][slot["leaf_index"]]
            if source["kind"] != "USER_INPUT":
                continue
            value = leaves[source["user_leaf"]]
            alias = state.aliases[slot["object_id"]]
            if alias in copied:
                continue
            copied.add(alias)
            raw = torch.empty(0, dtype=torch.uint8).set_(
                value.untyped_storage(), 0, (value.untyped_storage().nbytes(),), (1,)
            )
            state.host(alias).copy_(raw)
