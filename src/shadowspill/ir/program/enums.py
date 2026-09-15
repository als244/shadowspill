"""The vocabularies a program speaks: roles, persistence, residency, resources."""

from __future__ import annotations

from enum import StrEnum

from shadowspill.schema import artifact_schema

PROGRAM_SCHEMA = artifact_schema("program")


class ObjectRole(StrEnum):
    INPUT = "input"
    PARAMETER = "parameter"
    BUFFER = "buffer"
    ACTIVATION = "activation"
    GRADIENT = "gradient"
    OPTIMIZER_STATE = "optimizer_state"
    OUTPUT = "output"
    OTHER = "other"
    CONTROL = "control"


class Persistence(StrEnum):
    STEP = "step"
    RUN = "run"
    CHECKPOINT = "checkpoint"


class SharedResidencyPolicy(StrEnum):
    """Runtime-global residency and mutation policy for one storage root."""

    SHARED_READ_ONLY = "shared_read_only"
    SHARED_WRITABLE_CAUSAL = "shared_writable_causal"
    SHARED_WRITABLE_UNORDERED = "shared_writable_unordered"


class ResourceKind(StrEnum):
    COMPUTE = "compute"
    COMMUNICATION = "communication"
    CONTROL = "control"
