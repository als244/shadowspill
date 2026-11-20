"""What one task argument is.

`TaskInputRole` is the vocabulary: a task argument is an activation, a parameter,
a buffer, a constant, or a piece of optimizer state.
"""

from __future__ import annotations

from enum import StrEnum

#: The policy the representative values were generated under. A profile recorded
#: under another policy is not comparable with one recorded under this.
REPRESENTATIVE_VALUE_POLICY = "shadowspill.task-values/v6"


class TaskInputRole(StrEnum):
    """What one argument of a compiled task is.

    Attributes
    ----------
    PARAMETER : str
        A trained parameter.
    BUFFER : str
        Model state that is not trained.
    CONSTANT : str
        A constant captured with the task.
    USER_INPUT : str
        A value the caller supplies.
    CONTROL : str
        Integer or boolean data that steers the computation, such as indices or
        a mask; profiling needs its real value.
    STATISTIC : str
        Integer data from an earlier stage that the task uses only in
        arithmetic, never to steer the computation.
    ACTIVATION : str
        A value an earlier stage produced.
    RESIDUAL : str
        A value a forward saved for its backward.
    TANGENT : str
        The gradient a backward receives for one of its forward's outputs.
    GRADIENT : str
        A gradient a task adds into or an optimizer reads.
    OPTIMIZER_STATE : str
        State the optimizer keeps between updates.
    OPTIMIZER_HYPERPARAMETER : str
        A value the optimizer reads, such as its learning rate.
    ANONYMOUS : str
        An argument whose source is not known.
    """

    PARAMETER = "parameter"
    BUFFER = "buffer"
    CONSTANT = "constant"
    USER_INPUT = "user_input"
    CONTROL = "control"
    STATISTIC = "statistic"
    ACTIVATION = "activation"
    RESIDUAL = "residual"
    TANGENT = "tangent"
    GRADIENT = "gradient"
    OPTIMIZER_STATE = "optimizer_state"
    OPTIMIZER_HYPERPARAMETER = "optimizer_hyperparameter"
    ANONYMOUS = "anonymous"


__all__ = [
    "REPRESENTATIVE_VALUE_POLICY",
    "TaskInputRole",
]
