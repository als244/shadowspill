"""Bind a caller's input axes to an explicit task's physical tensor ports."""

import torch

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import GraphArtifact

from .geometry import ChunkAxis
from .policy import AxisChunks


def resolve_input_axes(
    artifact: GraphArtifact, config: AxisChunks
) -> tuple[tuple[ChunkAxis | None, ...], int]:
    mapping = artifact.input_components or tuple(
        (i, ()) for i in artifact.tensor_argument_positions
    )
    axes: list[ChunkAxis | None] = []
    lengths = set()
    seen = set()
    for (position, path), value in zip(
        mapping, artifact.example_arguments, strict=True
    ):
        selected = config.inputs.get(position)
        if selected is None:
            axes.append(None)
            continue
        axis = ChunkAxis(selected) if isinstance(selected, int) else selected
        dim = axis.dim
        if (
            path
            or not isinstance(value, torch.Tensor)
            or not -value.ndim <= dim < value.ndim
        ):
            raise CaptureError(f"invalid chunk dimension for stage input {position}")
        dim %= value.ndim
        factor = axis.before * axis.after
        if int(value.shape[dim]) % factor:
            raise CaptureError(f"chunk factors do not divide stage input {position}")
        lengths.add(int(value.shape[dim]) // factor)
        axes.append(ChunkAxis(dim, axis.before, axis.after))
        seen.add(position)
    if seen != set(config.inputs) or len(lengths) != 1 or 0 in lengths:
        raise CaptureError(
            "sliced stage inputs must exist and have the same positive axis extent"
        )
    return tuple(axes), lengths.pop()
