"""Accumulated-training planning, one phase per module.

`capture` exports the objective and partitions it on fake tensors; `materialize`
registers the model's state with the runtime and captures the optimizer once;
`profile` compiles and measures every structurally unique task; `programs`
lowers the canonical programs under one ordering and derives their admission
facts; `plan` searches them and certifies their layouts; `admit` compiles the
selected tasks, admits the plans physically and publishes the callable, with
`report` writing its plan report. `build` composes the phases into one training
plan; `steps` builds one `StepProgram` per ordering for the geometry search,
through the step archive. Each phase's outputs are the artifacts in
`planning/artifacts.py`.
"""

from shadowspill.pytorch.planning.training.plan import plan_training_programs

from ..artifacts import (
    TrainingCaptureArtifacts,
    TrainingExecutableArtifacts,
    TrainingMaterializationArtifacts,
    TrainingProfileArtifacts,
    TrainingProgramArtifacts,
)
from .admit import admit_training_plan, compile_selected_training_tasks
from .build import build_training
from .capture import capture_training_graphs
from .materialize import materialize_training_state, rollback_training_materialization
from .profile import profile_training_tasks
from .programs import build_training_programs
from .steps import make_training_programs

__all__ = [
    "TrainingCaptureArtifacts",
    "TrainingExecutableArtifacts",
    "TrainingMaterializationArtifacts",
    "TrainingProfileArtifacts",
    "TrainingProgramArtifacts",
    "admit_training_plan",
    "build_training",
    "build_training_programs",
    "capture_training_graphs",
    "compile_selected_training_tasks",
    "make_training_programs",
    "materialize_training_state",
    "plan_training_programs",
    "profile_training_tasks",
    "rollback_training_materialization",
]
