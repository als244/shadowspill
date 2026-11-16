"""Keep functional forward values while discarding selected buffer writebacks."""

from collections.abc import Sequence
from dataclasses import replace

from torch import nn
from torch.export.graph_signature import OutputKind

from shadowspill.errors import CaptureError
from shadowspill.pytorch.representations import tensor_components

from .export import ExportCapture


def preserved_buffer_names(model: nn.Module, names: Sequence[str]) -> tuple[str, ...]:
    """Resolve whole buffer aliases, rejecting parameter-backed storage."""
    buffers = dict(model.named_buffers(remove_duplicate=False))
    if isinstance(names, str) or any(
        not isinstance(name, str) or name not in buffers for name in names
    ):
        raise CaptureError("preserved buffers must name registered model buffers")
    storage = {
        name: {leaf.untyped_storage()._cdata for _, leaf in tensor_components(value)}
        for name, value in buffers.items()
    }
    selected = set().union(*(storage[name] for name in names))
    parameters = {
        leaf.untyped_storage()._cdata
        for parameter in model.parameters()
        for _, leaf in tensor_components(parameter)
    }
    if selected & parameters:
        raise CaptureError("preserved buffers must not alias model parameters")
    return tuple(name for name, aliases in storage.items() if aliases & selected)


def preserve_buffers(capture: ExportCapture, names: Sequence[str]) -> ExportCapture:
    """Remove only declared buffer mutation outputs from a functional export.

    Intermediate updated values (and returned views of them) still participate
    in the original computation. They become ordinary planned temporaries;
    nothing writes them back to the registered buffer. Other effects retain
    their contracts so the usual operation validation can reject them.
    """
    if not names:
        return capture
    exported = capture.exported_program
    signature = exported.graph_signature
    # The complete model validated these names before export. A selected
    # operation may not read or mutate every declared buffer.
    omitted = {
        mutation.output_index
        for mutation in capture.mutations
        if mutation.kind is OutputKind.BUFFER_MUTATION and mutation.target in names
    }
    if not omitted:
        return capture
    output = next(node for node in exported.graph.nodes if node.op == "output")
    indices = {
        old: new
        for new, old in enumerate(
            index
            for index in range(len(signature.output_specs))
            if index not in omitted
        )
    }
    output.args = (tuple(output.args[0][index] for index in indices),)
    signature.output_specs[:] = [signature.output_specs[index] for index in indices]
    exported.graph.eliminate_dead_code()
    exported.graph.lint()
    exported.graph_module.recompile()
    return replace(
        capture,
        user_output_indices=tuple(
            indices[index] for index in capture.user_output_indices
        ),
        mutations=tuple(
            replace(mutation, output_index=indices[mutation.output_index])
            for mutation in capture.mutations
            if mutation.output_index not in omitted
        ),
    )
