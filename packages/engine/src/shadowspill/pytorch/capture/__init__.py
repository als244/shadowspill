"""Offline PyTorch graph capture and semantic storage contracts."""

from .aot import capture_graph_pair
from .artifacts import (
    AotGraphPair,
    GraphArtifact,
    ObjectiveSchema,
    TaskInputProvenance,
    TensorGeometry,
    capture_forward_stage_artifacts,
    capture_objective_schema,
    normalize_objective_result,
)
from .export import (
    ExportCapture,
    ExportMutation,
    capture_forward,
    export_capture_digest,
    inference_artifact,
)
from .fake import fake_device_inputs, fake_device_model
from .objective import (
    TrainingCapture,
    TrainingObjectiveCapture,
    capture_training,
    capture_training_objective,
)
from .storage import (
    ExplicitMutation,
    MutationBinding,
    OutputView,
    StorageRoot,
    StorageRootKind,
    TaskStorageContract,
    capture_task_storage_contract,
)

__all__ = [
    "AotGraphPair",
    "ExplicitMutation",
    "ExportCapture",
    "ExportMutation",
    "GraphArtifact",
    "MutationBinding",
    "ObjectiveSchema",
    "OutputView",
    "StorageRoot",
    "StorageRootKind",
    "TaskInputProvenance",
    "TaskStorageContract",
    "TensorGeometry",
    "TrainingCapture",
    "TrainingObjectiveCapture",
    "capture_forward",
    "capture_forward_stage_artifacts",
    "capture_graph_pair",
    "capture_objective_schema",
    "capture_task_storage_contract",
    "capture_training",
    "capture_training_objective",
    "export_capture_digest",
    "fake_device_inputs",
    "fake_device_model",
    "inference_artifact",
    "normalize_objective_result",
]
