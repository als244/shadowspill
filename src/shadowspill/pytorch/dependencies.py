"""Portable registration dependencies for Python-defined custom operators."""

from __future__ import annotations

import hashlib
import importlib
import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from torch._library.custom_ops import OPDEFS, CustomOpDef
from torch.utils._pytree import tree_leaves


def _module_digest(name: str) -> str:
    module = importlib.import_module(name)
    filename = getattr(module, "__file__", None)
    if filename is None:
        raise ValueError(f"operator registration module has no saved source: {name}")
    return hashlib.sha256(Path(filename).read_bytes()).hexdigest()


def _check_registration(definition: CustomOpDef, origin: str) -> None:
    if definition._disabled_kernel:
        raise ValueError("temporarily disabled custom kernels cannot be saved")
    for wrapped in definition._backend_fns.values():
        kernel = inspect.getclosurevars(inspect.unwrap(wrapped)).nonlocals.get("fn")
        if getattr(kernel, "__module__", None) != origin:
            raise ValueError(
                f"{definition._qualname} has a separately registered kernel; "
                "it needs an explicit provider initializer"
            )


def operator_dependencies(graph: torch.fx.GraphModule) -> tuple[dict[str, str], ...]:
    """Describe non-ATen ops without serializing live Python/module objects.

    Ordinary custom_op definitions carry their importable defining module.
    C-extension/dynamic registrations need an explicit provider binding, which the
    initial single-GPU path deliberately refuses instead of silently depending
    on imports or resources from the build process.
    """
    dependencies = {}
    for module in graph.modules():
        if not isinstance(module, torch.fx.GraphModule):
            continue
        # Functionalized/effectful calls can carry an op as an argument to a
        # higher-order operator rather than as the direct node target.
        operations = (
            value
            for node in module.graph.nodes
            for value in tree_leaves((node.target, node.args, node.kwargs))
            if isinstance(value, torch._ops.OpOverload)
        )
        for op in operations:
            name = op._schema.name
            if op.namespace in {"aten", "prims", "inductor", "profiler"}:
                continue
            definition = OPDEFS.get(name)
            origin = None if definition is None else definition._init_fn.__module__
            if not origin or origin == "__main__":
                raise ValueError(
                    f"{name} needs an importable operator/provider registration; "
                    "the saved task cannot rely on the model process"
                )
            assert definition is not None
            _check_registration(definition, origin)
            dependencies[name] = {
                "operator": name,
                "schema": str(op._schema),
                "module": origin,
                "module_sha256": _module_digest(origin),
            }
    return tuple(dependencies[key] for key in sorted(dependencies))


def restore_dependencies(dependencies: Sequence[Mapping[str, str]]) -> None:
    """Import saved registrations and reject different implementations/schemas."""
    for dependency in dependencies:
        name = dependency["operator"]
        if _module_digest(dependency["module"]) != dependency["module_sha256"]:
            raise ValueError(f"operator registration source differs: {name}")
        definition = OPDEFS.get(name)
        if (
            definition is None
            or definition._init_fn.__module__ != dependency["module"]
            or str(definition._opoverload._schema) != dependency["schema"]
        ):
            raise ValueError(f"operator registration cannot be restored: {name}")
        _check_registration(definition, dependency["module"])
