"""What one task argument is, and the provenance of the value profiled for it.

`TaskInputRole` is the vocabulary: a task argument is an activation, a parameter,
a buffer, a constant, or a piece of optimizer state. `RepresentativeInputSummary`
is what profiling recorded about the value it materialized for one argument --
its geometry and provenance, never its contents, so a profile artifact says what
was measured without carrying any data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

#: The policy the representative values were generated under. A profile recorded
#: under an older policy is not comparable with one recorded under this.
REPRESENTATIVE_VALUE_POLICY = "shadowspill.task-values/v5"


class TaskInputRole(StrEnum):
    """Semantic source of one explicit compiled-task argument."""

    PARAMETER = "parameter"
    BUFFER = "buffer"
    CONSTANT = "constant"
    USER_INPUT = "user_input"
    CONTROL = "control"
    ACTIVATION = "activation"
    RESIDUAL = "residual"
    TANGENT = "tangent"
    GRADIENT = "gradient"
    OPTIMIZER_STATE = "optimizer_state"
    OPTIMIZER_HYPERPARAMETER = "optimizer_hyperparameter"
    ANONYMOUS = "anonymous"


@dataclass(frozen=True, slots=True)
class RepresentativeInputSummary:
    """Content-free provenance for one materialized task argument."""

    position: int
    role: TaskInputRole
    source: str | None
    value_policy: str
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    alias_group: int
    consumer_targets: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "position": self.position,
            "role": self.role.value,
            "source": self.source,
            "value_policy": self.value_policy,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "stride": list(self.stride),
            "storage_offset": self.storage_offset,
            "alias_group": self.alias_group,
            "consumer_targets": list(self.consumer_targets),
        }

    @classmethod
    def from_dict(cls, value: object) -> RepresentativeInputSummary:
        if not isinstance(value, dict):
            raise ValueError("representative input summary must be an object")
        try:
            return cls(
                position=int(value["position"]),
                role=TaskInputRole(str(value["role"])),
                source=(None if value["source"] is None else str(value["source"])),
                value_policy=str(value["value_policy"]),
                dtype=str(value["dtype"]),
                shape=tuple(int(item) for item in value["shape"]),
                stride=tuple(int(item) for item in value["stride"]),
                storage_offset=int(value["storage_offset"]),
                alias_group=int(value["alias_group"]),
                consumer_targets=tuple(str(item) for item in value["consumer_targets"]),
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                "representative input summary has an invalid schema"
            ) from exc


__all__ = [
    "REPRESENTATIVE_VALUE_POLICY",
    "RepresentativeInputSummary",
    "TaskInputRole",
]
