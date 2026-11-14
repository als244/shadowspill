"""Read scheduled buffer lifetimes; never guess inside opaque provider calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch._inductor.codegen import wrapper
from torch._inductor.graph import GraphLowering
from torch._inductor.virtualized import V


@dataclass
class _Buffers:
    graph: GraphLowering
    allocations: list[dict[str, Any]] = field(default_factory=list)
    names: dict[str, int] = field(default_factory=dict)
    unresolved: set[str] = field(default_factory=set)

    def allocate(self, node: Any, index: int) -> None:
        name = node.get_name()
        if name in self.names:
            self.unresolved.add(f"repeated allocation: {name}")
            return
        try:
            size = int(self.graph.get_allocation_storage_size(node))
            size *= torch.empty(
                (), device="meta", dtype=node.get_dtype()
            ).element_size()
        except (AttributeError, NotImplementedError, TypeError, ValueError):
            self.unresolved.add(f"allocation extent: {name}")
            return
        if node.get_device().type != "cuda":
            return
        self.names[name] = len(self.allocations)
        self.allocations.append(
            dict(
                names=[name], requested_bytes=size, begin=index, end=None, output=False
            )
        )

    def reuse(self, before: str, after: str, *, delete_old: bool) -> None:
        if before not in self.names:
            # Input aliases do not allocate task workspace.
            return
        ordinal = self.names[before]
        self.names[after] = ordinal
        self.allocations[ordinal]["names"].append(after)
        if delete_old and before != after:
            del self.names[before]

    def free(self, name: str, index: int) -> None:
        ordinal = self.names.pop(name, None)
        if ordinal is not None and ordinal not in self.names.values():
            self.allocations[ordinal]["end"] = index


def compiler_memory(graph: GraphLowering) -> dict[str, Any]:
    """Return static evidence and explicit gaps, independent of CUDA allocation.

    Sizes are compiler-requested bytes. This is not an allocator trace: provider
    scratch, persistent library resources, and physical rounding still require
    evidence before execution. The plain records also support CPU inspection.
    """
    with V.set_graph_handler(graph):
        return _compiler_memory(graph)


def _compiler_memory(graph: GraphLowering) -> dict[str, Any]:
    buffers = _Buffers(graph)
    code = graph.wrapper_code
    if code is None:
        return {"allocations": [], "unresolved": ["missing compiler wrapper"]}
    lines = code.lines
    for index, line in enumerate(lines):
        if isinstance(line, wrapper.AllocateLine):
            buffers.allocate(line.node, index)
        elif isinstance(line, wrapper.ReuseLine):
            buffers.reuse(
                line.node.get_name(),
                line.reused_as.get_name(),
                delete_old=line.delete_old,
            )
        elif isinstance(line, wrapper.ReinterpretLine):
            buffers.reuse(
                line.node.get_name(), line.reused_as.get_name(), delete_old=False
            )
        elif isinstance(line, wrapper.FreeIfNotReusedLine):
            if not line.is_reused:
                buffers.free(line.node.get_name(), index)
        elif isinstance(line, wrapper.FreeLine):
            buffers.free(line.node.get_name(), index)
        elif isinstance(line, wrapper.KernelCallLine):
            if not line.triton:
                buffers.unresolved.add(f"kernel scratch: {line.kernel_name}")
        elif isinstance(
            line,
            (
                wrapper.ExternKernelAllocLine,
                wrapper.ExternKernelOutLine,
                wrapper.ExternKernelMultiOutLine,
            ),
        ):
            buffers.unresolved.add(f"provider scratch: {line.node.get_kernel_name()}")
        elif isinstance(
            line,
            (
                wrapper.NullLine,
                wrapper.KernelDefinitionLine,
                wrapper.CommentLine,
                wrapper.AssertSizeStrideLine,
                wrapper.EnterDeviceContextManagerLine,
                wrapper.ExitDeviceContextManagerLine,
            ),
        ):
            pass
        else:
            # Unsupported control flow, private streams, fallback output
            # allocations, and raw wrapper statements need runtime evidence.
            buffers.unresolved.add(f"wrapper memory: {type(line).__name__}")
    for output in graph.graph_outputs or ():
        if not output.has_tensor_output():
            continue
        ordinal = buffers.names.get(output.get_name())
        if ordinal is not None:
            buffers.allocations[ordinal]["output"] = True
    events: list[tuple[int, int]] = []
    for record in buffers.allocations:
        if record["end"] is None:
            record["end"] = len(lines)
        if not record["output"]:
            events.extend(
                (
                    (record["begin"], record["requested_bytes"]),
                    (record["end"], -record["requested_bytes"]),
                )
            )
    live = peak = 0
    for _, change in sorted(events):
        live += change
        peak = max(peak, live)
    return {
        "allocations": buffers.allocations,
        "known_workspace_bytes": peak,
        "unresolved": sorted(buffers.unresolved),
    }
