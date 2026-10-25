"""Captured graphs, tensor metadata and stage/output schemas."""

from shadowspill.pytorch.contracts import normalize_objective_result

from .geometry import TaskInputProvenance, TensorGeometry
from .graph import GraphArtifact
from .inference import capture_forward_stage_artifacts, capture_inference_artifact
from .schema import AotGraphPair, ObjectiveSchema, capture_objective_schema

__all__ = [
    "AotGraphPair",
    "GraphArtifact",
    "ObjectiveSchema",
    "TaskInputProvenance",
    "TensorGeometry",
    "capture_forward_stage_artifacts",
    "capture_inference_artifact",
    "capture_objective_schema",
    "normalize_objective_result",
]
