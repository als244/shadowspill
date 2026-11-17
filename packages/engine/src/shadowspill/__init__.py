"""Framework-neutral memory planning and transparent execution support."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import Any

try:
    __version__ = version("shadowspill")
except PackageNotFoundError:  # pragma: no cover - source tree without install
    __version__ = "0.1.0.dev0"

_PUBLIC = {
    "build_forward": "build.forward",
    "build_forward_backward": "build.training",
    "build_step": "build.step",
    "Program": "artifacts",
    "Profiles": "artifacts",
    "ProfiledProgram": "artifacts",
    "Plan": "artifacts",
    "TaskProfile": "artifacts",
    "SweepResult": "artifacts",
    "apply_profiles": "profiling.apply",
    "measure_program": "profiling.measure",
    "ProfilingOptions": "task.profiling",
    "TaskProfileCache": "profiling.cache",
    "OverlapProfileCache": "profiling.overlap_cache",
    "measure_overlap": "profiling.overlap",
    "MemoryBudgets": "artifacts.budgets",
    "PoolMemoryProfile": "artifacts.pool_memory",
    "measure_pool_memory": "profiling.pool_memory",
    "CommunicationInfo": "communication",
    "measure_communication": "communication.measure",
    "plan_program": "planning.plan",
    "optimizer_program": "planning.training.operations",
    "backward_program": "planning.training.backward",
    "instantiate_examples": "planning.training.examples",
    "sweep_programs": "planning.sweep",
    "PlanningInputs": "training.preparation.local_planning",
    "write_report": "reporting",
    "Executor": "execution",
    "TrainingResult": "execution",
    "TensorChunk": "artifacts.tensors",
    "save_checkpoint": "checkpoint",
    "load_checkpoint": "checkpoint",
    "Trainer": "training.trainer",
    "Parallelism": "parallelism",
    "ParameterStorage": "parallelism",
    "Replicated": "parallelism",
    "Sharded": "parallelism",
    "GroupRef": "parallelism",
    "Resource": "resources",
    "OperatorPreparation": "preparation",
}


def __getattr__(name: str) -> Any:
    if name not in _PUBLIC:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_PUBLIC[name]}", __name__), name)
    globals()[name] = value
    return value


__all__ = ["__version__", *_PUBLIC]
