"""Distinguish static wrapper geometry from unresolved value-dependent behavior.

This describes allocation invariance, not byte sizes or physical admission.
Opaque calls remain explicit requirements for source-pinned provider contracts.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from torch._inductor.codegen import wrapper
from torch._inductor.utils import LineContext

_SCOPE = (
    "fixed tensor geometry/layout and captured non-tensor arguments; all tensor values"
)


def wrapper_allocation_behavior(
    lines: Sequence[Any], *, geometry_gaps: Iterable[str] = ()
) -> dict[str, Any]:
    """Inspect compiler instructions, leaving every unknown behavior unresolved."""
    unknown, opaque = set(geometry_gaps), set()
    for line in lines:
        if isinstance(
            line,
            (
                wrapper.ExternKernelAllocLine,
                wrapper.ExternKernelOutLine,
                wrapper.ExternKernelMultiOutLine,
            ),
        ):
            opaque.add(line.node.get_kernel_name())
        elif isinstance(line, wrapper.KernelCallLine):
            if not line.triton:
                opaque.add(line.kernel_name)
        elif isinstance(line, wrapper.UnbackedSymbolDefsLine):
            if line.unbacked_bindings:
                unknown.add("unbacked values can change generated geometry")
        elif isinstance(line, wrapper.MultiOutputLine):
            if not all(
                kind in (tuple, list, dict) and type(index) in (int, str)
                for kind, index in line.indices
            ):
                unknown.add("unresolved provider output selection")
        elif isinstance(
            line,
            (
                wrapper.AllocateLine,
                wrapper.ReuseLine,
                wrapper.ReinterpretLine,
                wrapper.FreeLine,
                wrapper.FreeIfNotReusedLine,
                wrapper.NullLine,
                wrapper.KernelDefinitionLine,
                wrapper.CommentLine,
                wrapper.AssertSizeStrideLine,
                wrapper.EnterDeviceContextManagerLine,
                wrapper.ExitDeviceContextManagerLine,
                LineContext,
            ),
        ):
            # Allocation extents were checked by the compiler memory reader.
            # CUDA context/retained stream state is separately measured at startup.
            pass
        elif isinstance(line, str) and _static_statement(line):
            pass
        else:
            unknown.add(f"wrapper behavior: {type(line).__name__}")
    return {
        "version": 1,
        "scope": _SCOPE,
        "opaque_calls": sorted(opaque),
        "unresolved": sorted(unknown),
    }


def _static_statement(source: str) -> bool:
    try:
        statements = ast.parse(source.strip()).body
    except SyntaxError:
        return False
    for statement in statements:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            if (
                isinstance(call.func, ast.Name)
                and call.func.id == "assert_alignment"
                and not call.keywords
                and len(call.args) == 3
                and isinstance(call.args[0], ast.Name)
                and isinstance(call.args[1], ast.Constant)
                and type(call.args[1].value) is int
                and isinstance(call.args[2], ast.Constant)
                and isinstance(call.args[2].value, str)
            ):
                # Generated output-address guard; it never inspects tensor data.
                continue
        if isinstance(statement, ast.Delete):
            if not all(isinstance(target, ast.Name) for target in statement.targets):
                return False
            continue
        if not isinstance(statement, ast.Assign) or not all(
            isinstance(target, ast.Name) for target in statement.targets
        ):
            return False
        value = statement.value
        if isinstance(value, (ast.Name, ast.Constant)):
            continue
        if not isinstance(value, ast.Call) or value.keywords:
            return False
        if not isinstance(value.func, ast.Name) or len(value.args) != 1:
            return False
        argument = value.args[0]
        if value.func.id == "copy_if_misaligned" and isinstance(argument, ast.Name):
            # Pointer alignment is part of the captured physical input layout;
            # this branch cannot inspect tensor data or change with its values.
            continue
        if (
            value.func.id == "get_raw_stream"
            and isinstance(argument, ast.Constant)
            and type(argument.value) is int
        ):
            continue
        return False
    return True


def composed_allocation_behavior(
    children: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """A fixed row/region traversal inherits every child's unresolved behavior."""
    unknown, opaque = set(), set()
    for child in children:
        record = child.get("allocation_behavior")
        if not isinstance(record, Mapping) or record.get("version") != 1:
            unknown.add("composed child has no allocation behavior description")
            continue
        if record.get("scope") != _SCOPE:
            unknown.add("composed child has incompatible allocation behavior scope")
        unknown.update(record["unresolved"])
        opaque.update(record["opaque_calls"])
    return {
        "version": 1,
        "scope": _SCOPE,
        "opaque_calls": sorted(opaque),
        "unresolved": sorted(unknown),
    }
