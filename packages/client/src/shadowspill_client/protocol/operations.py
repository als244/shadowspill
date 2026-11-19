"""Portable identities for prepared operation/output/mode combinations."""

from dataclasses import dataclass
from typing import Literal

from .._records import Record

type ComputeOperation = Literal["step", "forward_backward", "forward", "optim_step"]
type Mode = Literal["training", "evaluation"]


@dataclass(frozen=True)
class OperationKey(Record):
    """A compute operation as prepared: the operation, its output set and mode.

    Parameters
    ----------
    operation : str
        ``"step"``, ``"forward_backward"``, ``"forward"`` or ``"optim_step"``.
    outputs : str or None
        Name of the objective's output set the operation produces, such as
        ``"metrics"``; None for ``"optim_step"``, which produces none.
    mode : str, default "training"
        ``"training"``, or ``"evaluation"`` for ``"forward"`` only.

    Raises
    ------
    ValueError
        If the operation is unknown, ``"evaluation"`` is given for another
        operation than ``"forward"``, or ``outputs`` is set for
        ``"optim_step"`` or missing for another operation.
    """

    operation: ComputeOperation
    outputs: str | None
    mode: Mode = "training"

    def __post_init__(self) -> None:
        if self.operation not in ("step", "forward_backward", "forward", "optim_step"):
            raise ValueError("unknown compute operation")
        if self.mode not in ("training", "evaluation") or (
            self.operation != "forward" and self.mode != "training"
        ):
            raise ValueError("only forward supports evaluation mode")
        if self.operation == "optim_step":
            if self.outputs is not None:
                raise ValueError("optimizer-only execution has no objective outputs")
        elif not isinstance(self.outputs, str) or not self.outputs:
            raise ValueError("compute operation requires an output set")
