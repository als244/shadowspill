"""Bind declared operations to physical inputs and the existing capture frontend."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from shadowspill.training.definitions.registry import ResolvedModel
from shadowspill.training.inputs.adapters import AdaptedInputs, InputBinding
from shadowspill.training.inputs.validation import combined_schema
from shadowspill.training.outputs import OutputLayout
from shadowspill_client.protocol.controls import ControlSchema, Scalar

from .operations import OperationContract
from .scalars import control_values


def control_tensors(
    schema: ControlSchema, values: Mapping[str, Scalar]
) -> dict[str, Any]:
    """Only the build/staging worker materializes these bounded scalar tensors."""
    import torch

    return {
        name: torch.tensor(
            value,
            dtype={"float": torch.float32, "int": torch.int64, "bool": torch.bool}[
                schema.fields[name].dtype
            ],
            device="cpu",
        )
        for name, value in control_values(schema, values).items()
    }


class OperationBinding:
    def __init__(
        self,
        model: ResolvedModel,
        contract: OperationContract,
        inputs: InputBinding,
        outputs: OutputLayout,
        buffer: int,
    ) -> None:
        layout = inputs.layout
        schema = combined_schema(model.inputs, contract.objective.inputs)
        if layout.schema != schema or outputs.schema != schema:
            raise ValueError("operation schemas differ from the resolved definitions")
        if (
            outputs.outputs != contract.outputs
            or outputs.placement != layout.placement
            or type(buffer) is not int
            or buffer < 0
            or buffer >= len(outputs.schedule.buffers)
            or outputs.schedule.buffers[buffer] != layout.buffer
        ):
            raise ValueError("operation output layout differs from its physical inputs")
        partition = layout.placement.partition_index(layout.logical_rank)
        start = sum(len(p) for p in outputs.schedule.partitions[:partition])
        if not start <= buffer < start + len(outputs.schedule.partitions[partition]):
            raise ValueError("operation buffer belongs to another input partition")
        if set(layout.buffer.fields) != set(contract.required_paths):
            raise ValueError("physical inputs must match the selected required fields")
        if (
            not inputs.adapter.padding_safe
            or not (
                contract.objective.definition.padding_safe
                or contract.objective.definition.physical_batch_loss
            )
        ):
            raise ValueError("model adapter and objective must declare padding safety")
        self.model, self.contract = model, contract
        self.inputs, self.outputs, self.buffer = inputs, outputs, buffer
        self.logical_rank = layout.logical_rank
        self.contributes_loss = layout.placement.contributes_loss(self.logical_rank)
        self.computation = contract.objective.computation(contract.key)
        self.control_names = tuple(contract.controls.fields)
        self.metrics = tuple(contract.outputs.metrics)
        self.tensor_dtypes = {
            name: output.dtype for name, output in contract.outputs.tensors.items()
        }
        self.tensor_shapes = {
            name: outputs.worker_shape(name, buffer, self.logical_rank)
            for name in contract.outputs.tensors
        }

    def arguments(
        self,
        adapted: AdaptedInputs,
        loss_params: Mapping[str, Scalar] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        if (
            adapted.owner.logical_rank != self.logical_rank
            or adapted.fields.keys() != self.inputs.shapes.keys()
            or adapted.dtypes != self.inputs.dtypes
            or any(
                tuple(value.shape) != self.inputs.shapes[path]
                for path, value in adapted.fields.items()
            )
        ):
            raise ValueError("adapted fields differ from the prepared operation")
        model_inputs, loss_inputs = adapted.torch_inputs()
        # Sort recursively, so user-leaf identities never depend on adapter order.
        return (
            _ordered(model_inputs),
            _ordered(loss_inputs),
            control_tensors(
                self.contract.controls, self.contract.loss_params(loss_params)
            ),
        )

    def _result(
        self,
        model: Any,
        inputs: Mapping[str, Any],
        loss_inputs: Mapping[str, Any],
        loss_params: Mapping[str, Any],
        *,
        forward: bool,
    ) -> tuple[Any, dict[str, Any]]:
        import torch

        from shadowspill.pytorch.contracts import normalize_objective_result

        if set(loss_params) != set(self.control_names):
            raise ValueError("objective controls differ from the prepared operation")
        value = self.computation(model, inputs, loss_inputs, loss_params)
        if forward and isinstance(value, Mapping):
            loss, values = None, value
        else:
            loss, values = normalize_objective_result(value, require_grad=False)
            if loss.is_complex():
                raise ValueError("training service requires a real scalar loss")
            if not self.contributes_loss:
                # Keep a differentiable graph on cooperating workers. Generic
                # placement decides ownership; never divide by worker count.
                loss = loss * 0
        if values is None:
            values = {}
        expected = set(self.metrics) | set(self.tensor_shapes)
        if not isinstance(values, Mapping) or set(values) != expected:
            raise ValueError("computation outputs differ from the declared output set")
        for name in self.metrics:
            metric = values[name]
            if (
                not isinstance(metric, torch.Tensor)
                or metric.shape != ()
                or metric.is_complex()
            ):
                raise ValueError(f"metric {name} must be a real scalar tensor")
        for name, shape in self.tensor_shapes.items():
            tensor = values[name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != shape
                or str(tensor.dtype)
                != "torch." + self.tensor_dtypes[name]
            ):
                raise ValueError(
                    f"tensor output {name} differs from its declared layout"
                )
        return loss, {name: values[name] for name in sorted(expected)}

    def objective(
        self, model: Any, inputs: Any, loss_inputs: Any, loss_params: Any
    ) -> Any:
        from shadowspill.pytorch.contracts import ObjectiveResult

        loss, values = self._result(
            model, inputs, loss_inputs, loss_params, forward=False
        )
        return ObjectiveResult(loss, values)

    def forward(
        self, model: Any, inputs: Any, loss_inputs: Any, loss_params: Any
    ) -> Any:
        return self._result(model, inputs, loss_inputs, loss_params, forward=True)

    def signature(self) -> dict[str, Any]:
        objective = self.contract.objective
        return {
            "version": 1,
            "operation": self.contract.key.to_dict(),
            "model": {
                "id": self.model.definition.id,
                "source": self.model.source.to_dict(),
                "config": dict(self.model.config),
                "trainable": None
                if self.model.settings is None or self.model.settings.trainable is None
                else list(self.model.settings.trainable),
                "mutable_buffers": list(self.model.definition.mutable_buffers),
            },
            "objective": {
                "id": objective.definition.id,
                "source": objective.source.to_dict(),
                "config": dict(objective.config),
                "padding_safe": objective.definition.padding_safe,
                "physical_batch_loss": objective.definition.physical_batch_loss,
                "participation": objective.participation.to_dict(),
                "reduction": objective.definition.reduction,
            },
            "inputs": self.inputs.signature(),
            "required_paths": [list(p) for p in self.contract.required_paths],
            "controls": self.contract.controls.to_dict(),
            "defaults": dict(self.contract.loss_params()),
            "outputs": self.contract.outputs.to_dict(),
            "output_layout": self.outputs.to_dict(),
            "buffer": self.buffer,
            "logical_rank": self.logical_rank,
            "contributes_loss": self.contributes_loss,
        }


def _ordered(tree: Mapping[str, Any]) -> dict[str, Any]:
    return {
        name: _ordered(value) if isinstance(value, Mapping) else value
        for name, value in sorted(tree.items())
    }
