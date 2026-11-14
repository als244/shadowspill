"""Exact output geometry for composed task boundaries."""

from dataclasses import replace
from typing import Any

import torch

from shadowspill.errors import CaptureError
from shadowspill.pytorch.capture.artifacts import GraphArtifact
from shadowspill.task.storage import StorageRootKind, TaskStorageContract


def materialized_storage(
    artifact: GraphArtifact, values: list[Any]
) -> TaskStorageContract:
    """Preserve absolute offsets expected by consumers of fresh partial views."""
    storage = artifact.storage_contract
    origins = {root.root_id: 0 for root in storage.roots}
    for view in storage.output_views:
        if storage.roots[view.root_id].kind is StorageRootKind.FRESH:
            value = values[view.leaf_index]
            origins[view.root_id] = (
                value.storage_offset() * value.element_size() - view.offset_bytes
            )
    return TaskStorageContract.build(
        tuple(
            replace(
                root, minimum_span_bytes=root.minimum_span_bytes + origins[root.root_id]
            )
            for root in storage.roots
        ),
        tuple(
            replace(view, offset_bytes=view.offset_bytes + origins[view.root_id])
            for view in storage.output_views
        ),
        storage.mutations,
    )


def boundary_metadata(artifact: GraphArtifact, values: list[Any]) -> dict[str, Any]:
    literals = []
    for index, value in enumerate(values):
        if not isinstance(value, torch.Tensor) and value is not None:
            if not isinstance(value, (str, bool, int, float)):
                raise CaptureError(
                    "execution region outputs require portable scalar literals"
                )
            literals.append((index, value))
    return {
        "storage": materialized_storage(artifact, values).to_dict(),
        "input_offsets": [x.storage_offset for x in artifact.tensor_inputs],
        "output_count": artifact.output_count,
        "output_literals": literals,
    }
