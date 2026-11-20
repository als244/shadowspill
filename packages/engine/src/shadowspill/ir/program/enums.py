"""The vocabularies a program speaks: roles, persistence, residency, resources."""

from __future__ import annotations

from enum import StrEnum

from shadowspill.schema import artifact_schema

PROGRAM_SCHEMA = artifact_schema("program")


class ObjectRole(StrEnum):
    """What an object holds, as the frontend labels it.

    Attributes
    ----------
    INPUT : str
        A value the caller supplies.
    PARAMETER : str
        A trained parameter.
    BUFFER : str
        Model state that is not trained.
    ACTIVATION : str
        A value one task produces for later tasks.
    GRADIENT : str
        A gradient.
    OPTIMIZER_STATE : str
        State the optimizer keeps between updates.
    OUTPUT : str
        A value the caller receives.
    OTHER : str
        Anything else; the default.
    CONTROL : str
        Integer or boolean data, such as indices or a mask.
    """

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
    """How long an object's value lasts, as the frontend labels it.

    Attributes
    ----------
    STEP : str
        One step; the default.
    RUN : str
        Every step of a run, without being saved in a checkpoint.
    CHECKPOINT : str
        Every step of a run, and saved in a checkpoint.
    """

    STEP = "step"
    RUN = "run"
    CHECKPOINT = "checkpoint"


class SharedResidencyPolicy(StrEnum):
    """How the runtime shares one alias group's device copy across plans.

    A group with a policy is on the device throughout, owned by the runtime
    rather than by any schedule: no memory action, schedule residency or
    retained-alias set may name it.

    Attributes
    ----------
    SHARED_READ_ONLY : str
        Plans share the copy, which tasks only read.
    SHARED_WRITABLE_CAUSAL : str
        Plans share the copy; every reader and writer acquires its current
        generation and waits for it to be ready. The only policy whose group
        may hold task outputs.
    SHARED_WRITABLE_UNORDERED : str
        Plans share the copy and tasks may update it in place, with no
        ordering between one plan's reads and another's writes.
    """

    SHARED_READ_ONLY = "shared_read_only"
    SHARED_WRITABLE_CAUSAL = "shared_writable_causal"
    SHARED_WRITABLE_UNORDERED = "shared_writable_unordered"


class ResourceKind(StrEnum):
    """The kind of queue a task runs on.

    Attributes
    ----------
    COMPUTE : str
        Computation on the device.
    COMMUNICATION : str
        A collective among devices.
    CONTROL : str
        A boundary that does no work, such as the start of a step.
    """

    COMPUTE = "compute"
    COMMUNICATION = "communication"
    CONTROL = "control"
