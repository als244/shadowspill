"""Portable identities for prepared operation/output/mode combinations."""

from dataclasses import dataclass
from typing import Literal

from .._records import Record

type ComputeOperation = Literal["step", "forward_backward", "forward", "optim_step"]
type Mode = Literal["training", "evaluation"]


@dataclass(frozen=True)
class OperationKey(Record):
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
