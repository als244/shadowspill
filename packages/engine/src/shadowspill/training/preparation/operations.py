"""CPU resolution of immutable operation/output contracts, before capture."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from shadowspill.training.definitions.registry import ResolvedObjective
from shadowspill.training.inputs.validation import Path
from shadowspill_client.config.operations import Operation
from shadowspill_client.protocol.controls import ControlSchema, Scalar
from shadowspill_client.protocol.definitions import OutputSet
from shadowspill_client.protocol.operations import OperationKey


def resolve_operations(
    objective: ResolvedObjective,
    operations: Mapping[str, Sequence[str | Operation]] | None,
    *,
    optimizer: bool,
) -> tuple[OperationKey, ...]:
    """Keep caller order/defaults and make optimizer dependencies inspectable."""
    selected = (
        {"step": [objective.definition.default_outputs]}
        if operations is None
        else operations
    )
    if not isinstance(selected, Mapping) or not selected:
        raise ValueError("request at least one compute operation")
    result: list[OperationKey] = []
    for operation, values in selected.items():
        if operation not in ("step", "forward_backward", "forward"):
            raise ValueError(f"unknown requested operation: {operation}")
        if operation == "step" and not optimizer:
            raise ValueError("step requires an optimizer")
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError("operation outputs must be a nonempty list")
        for value in values:
            choice = Operation(value) if isinstance(value, str) else value
            if not isinstance(choice, Operation):
                raise TypeError("operation entries must be output names or Operation")
            key = OperationKey(
                operation,  # type: ignore[arg-type]
                choice.outputs,
                choice.mode or ("evaluation" if operation == "forward" else "training"),
            )
            if key.outputs not in objective.outputs:
                raise ValueError(f"unknown output set: {key.outputs}")
            if key in result:
                raise ValueError("duplicate operation/output/mode combination")
            result.append(key)
    if optimizer and any(key.operation == "forward_backward" for key in result):
        result.append(OperationKey("optim_step", None))
    return tuple(result)


@dataclass(frozen=True)
class OperationContract:
    key: OperationKey
    objective: ResolvedObjective

    def __post_init__(self) -> None:
        if self.key.outputs not in self.objective.outputs:
            raise ValueError("operation requires a declared objective output set")

    @property
    def outputs(self) -> OutputSet:
        assert self.key.outputs is not None
        return self.objective.outputs[self.key.outputs].select(
            forward=self.key.operation == "forward"
        )

    @property
    def required_paths(self) -> tuple[Path, ...]:
        paths = {("inputs", *path) for path in self.outputs.model_inputs} | {
            ("loss_inputs", *path) for path in self.outputs.loss_inputs
        }
        participation = self.objective.definition.participation.path
        if self.key.operation != "forward" and participation:
            paths.add(("loss_inputs", *participation))
        return tuple(sorted(paths))

    @property
    def controls(self) -> ControlSchema:
        return ControlSchema(
            {
                name: self.objective.controls.fields[name]
                for name in sorted(self.outputs.controls)
            }
        )

    def loss_params(
        self, overrides: Mapping[str, Scalar] | None = None
    ) -> Mapping[str, Scalar]:
        return self.controls.resolve(
            defaults={
                name: self.objective.defaults[name] for name in self.controls.fields
            },
            overrides=overrides,
        )
