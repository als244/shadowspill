"""Named prepared outputs and explicit forward mode."""

from dataclasses import dataclass
from typing import Literal

from .._records import Record, text


@dataclass(frozen=True)
class Operation(Record):
    """An output set to prepare for a compute operation, with its model mode.

    It stands in for a plain output-set name in a run's ``operations``
    setting, as in ``{"forward": [Operation("logprobs", mode="training")]}``.

    Parameters
    ----------
    outputs : str
        Name of an output set the objective declares, such as ``"metrics"``
        or ``"logprobs"``.
    mode : str, optional
        Mode the model runs in: ``"evaluation"`` or ``"training"``. In
        ``"training"`` it may update the buffers it declares mutable;
        ``"evaluation"`` leaves all model state unchanged and is accepted only
        for ``forward``. None selects ``"evaluation"`` for ``forward`` and
        ``"training"`` otherwise.
    """

    outputs: str
    mode: Literal["evaluation", "training"] | None = None

    def __post_init__(self) -> None:
        text(self.outputs, "outputs")
        if self.mode not in (None, "evaluation", "training"):
            raise ValueError("operation mode must be evaluation or training")
