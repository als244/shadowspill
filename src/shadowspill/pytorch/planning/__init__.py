"""Composable high-level PyTorch planning boundaries."""

from shadowspill.planner.artifact_store import ArtifactStore

from .artifacts import (
    ForwardCaptureArtifacts,
    ForwardProfileArtifacts,
    ForwardProgramArtifacts,
    TrainingAdmissionArtifacts,
    TrainingCaptureArtifacts,
    TrainingExecutableArtifacts,
    TrainingMaterializationArtifacts,
    TrainingProfileArtifacts,
    TrainingProgramArtifacts,
    TrainingSelections,
)
from .common import PlanningTimer
from .forward import (
    admit_forward_plan,
    build_forward_program,
    capture_forward_graph,
    pressurefit_forward_program,
    profile_forward_tasks,
)
from .stores import PlanningStores, open_planning_stores
from .training import (
    admit_training_plan,
    build_training_programs,
    capture_training_graphs,
    compile_selected_training_tasks,
    materialize_training_state,
    pressurefit_training_programs,
    profile_training_tasks,
    rollback_training_materialization,
)

__all__ = [
    "ArtifactStore",
    "ForwardCaptureArtifacts",
    "ForwardProfileArtifacts",
    "ForwardProgramArtifacts",
    "PlanningStores",
    "PlanningTimer",
    "TrainingAdmissionArtifacts",
    "TrainingCaptureArtifacts",
    "TrainingExecutableArtifacts",
    "TrainingMaterializationArtifacts",
    "TrainingProfileArtifacts",
    "TrainingProgramArtifacts",
    "TrainingSelections",
    "admit_forward_plan",
    "admit_training_plan",
    "build_forward_program",
    "build_training_programs",
    "capture_forward_graph",
    "capture_training_graphs",
    "compile_selected_training_tasks",
    "materialize_training_state",
    "open_planning_stores",
    "pressurefit_forward_program",
    "pressurefit_training_programs",
    "profile_forward_tasks",
    "profile_training_tasks",
    "rollback_training_materialization",
]
