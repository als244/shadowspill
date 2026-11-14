"""Named prepared outputs and explicit forward mode."""

from dataclasses import dataclass
from typing import Literal

from .._records import Record, text


@dataclass(frozen=True)
class Operation(Record):
    outputs: str
    mode: Literal["evaluation", "training"] | None = None

    def __post_init__(self) -> None:
        text(self.outputs, "outputs")
        if self.mode not in (None, "evaluation", "training"):
            raise ValueError("operation mode must be evaluation or training")
