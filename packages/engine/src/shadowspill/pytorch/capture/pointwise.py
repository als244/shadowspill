"""Precision-bearing annotations on explicit pointwise graphs."""

from dataclasses import dataclass

from torch._functorch._aot_autograd.descriptors import (
    AOTOutput,
    BackwardTokenAOTOutput,
    DummyAOTOutput,
    ForwardTokenAOTOutput,
    SavedForBackwardsAOTOutput,
    SavedForBackwardsNoVcCheckAOTOutput,
)
from torch.fx import GraphModule, Node
from torch.utils._pytree import tree_leaves

from shadowspill.errors import CaptureError


def pointwise_precision_barrier(node: Node) -> bool:
    value = node.meta.get("low_precision_pointwise_barrier", False)
    if not isinstance(value, bool):
        raise CaptureError("FX low-precision pointwise barrier must be a boolean")
    return value


@dataclass(frozen=True, slots=True)
class SavedOutput:
    position: int
    index: int
    check_version: bool


def align_unlifted_output_descriptors(graph: GraphModule) -> None:
    """Keep AOT output metadata aligned after effect-token outputs are removed."""
    output = next(node for node in graph.graph.nodes if node.op == "output")
    descriptors = output.meta.get("desc")
    count = len(tree_leaves(output.args[0]))
    if descriptors is None or len(descriptors) == count:
        return
    # The pinned AOT unlift pass removes token values from the signature but
    # retains their descriptors. Only that explicit transformation is repaired.
    remaining = [
        descriptor
        for descriptor in descriptors
        if not isinstance(descriptor, (ForwardTokenAOTOutput, BackwardTokenAOTOutput))
    ]
    if len(remaining) != count or not any(
        str(node.target) == "prims._sink_tokens.default" for node in graph.graph.nodes
    ):
        raise CaptureError("AOT output precision descriptors do not match outputs")
    output.meta["desc"] = remaining


def saved_outputs(node: Node) -> tuple[SavedOutput, ...]:
    if node.op != "output":
        return ()
    descriptors = node.meta.get("desc")
    if descriptors is None:
        return ()
    outputs = tree_leaves(node.args[0])
    result = []
    for position, descriptor in enumerate(descriptors):
        if isinstance(
            descriptor,
            (SavedForBackwardsAOTOutput, SavedForBackwardsNoVcCheckAOTOutput),
        ):
            if position >= len(outputs):
                raise CaptureError("saved-output precision descriptor exceeds outputs")
            result.append(
                SavedOutput(
                    position,
                    descriptor.idx,
                    isinstance(descriptor, SavedForBackwardsAOTOutput),
                )
            )
    return tuple(result)


def restore_saved_outputs(node: Node, saved: tuple[SavedOutput, ...]) -> None:
    if not saved:
        return
    if node.op != "output":
        raise CaptureError("saved-output precision descriptors require an output node")
    descriptors: list[AOTOutput] = [
        DummyAOTOutput(i) for i, _ in enumerate(tree_leaves(node.args[0]))
    ]
    for entry in saved:
        if not 0 <= entry.position < len(descriptors):
            raise CaptureError("saved-output precision descriptor exceeds outputs")
        constructor = (
            SavedForBackwardsAOTOutput
            if entry.check_version
            else SavedForBackwardsNoVcCheckAOTOutput
        )
        descriptors[entry.position] = constructor(entry.index)
    node.meta["desc"] = descriptors
