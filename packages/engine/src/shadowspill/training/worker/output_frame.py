"""Fixed worker result frames derived on CPU from the admitted output contracts."""

from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from shadowspill.artifacts.examples import output_groups, single_forward
from shadowspill.artifacts.program import Program
from shadowspill.artifacts.tensors import public_tensor_geometry
from shadowspill.training.inputs.frames import (
    FRAME_CHUNK_BYTES,
    BufferWriter,
    InputFrame,
)
from shadowspill.training.preparation.capture.binding import OperationBinding
from shadowspill.training.preparation.planning.records import PreparedPlans
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef

if TYPE_CHECKING:
    import torch


class WorkerOutputFrame:
    """One saved output ABI; total file bytes and copy scratch are known up front.

    The existing tensor wire format is reused. Its inputs tree indexes worker
    buffers, each with a separate loss and named values branch. Global reductions
    and caller-order placement remain with the prepared OutputLayout/controller.
    """

    def __init__(self, program: Program, bindings: Sequence[OperationBinding]) -> None:
        if single_forward(program.bindings):
            raise ValueError("service output frames require grouped forward results")
        groups = output_groups(program.bindings)
        if len(groups) != len(bindings):
            raise ValueError("public outputs differ from the worker buffer schedule")
        geometry = public_tensor_geometry(program)
        shapes: dict[tuple[str, ...], tuple[int, ...]] = {}
        dtypes: dict[tuple[str, ...], str] = {}
        self.buffers = tuple(binding.buffer for binding in bindings)
        self.names = tuple(
            tuple(
                sorted(
                    set(binding.contract.outputs.metrics) | set(binding.tensor_shapes)
                )
            )
            for binding in bindings
        )
        self.losses = tuple(bool(group and group[0] is not None) for group in groups)
        self.forward = program.bindings["kind"] == "forward"
        for index, (group, binding, names) in enumerate(
            zip(groups, bindings, self.names, strict=True)
        ):
            if len(group) != 1 + len(names):
                raise ValueError("public tensor leaves differ from declared outputs")
            for position, (name, identity) in enumerate(
                zip(("loss", *names), group, strict=True)
            ):
                if identity is None:
                    if position:
                        raise ValueError(
                            "declared worker values must be tensor outputs"
                        )
                    continue
                view = geometry[identity]
                shape, dtype = (
                    tuple(view["shape"]),
                    view["dtype"].removeprefix("torch."),
                )
                path = (
                    ("inputs", str(index), "values", name)
                    if position
                    else ("inputs", str(index), "loss")
                )
                if not position or name in binding.contract.outputs.metrics:
                    if shape:
                        raise ValueError(
                            "worker loss and metrics must be scalar outputs"
                        )
                elif (
                    shape != binding.tensor_shapes[name]
                    or dtype != binding.contract.outputs.tensors[name].dtype
                ):
                    raise ValueError(
                        "saved tensor output differs from declared geometry"
                    )
                shapes[path], dtypes[path] = shape, dtype
        self.frame = InputFrame(shapes, dtypes)

    def fields(self, value: Any) -> Mapping[tuple[str, ...], "torch.Tensor"]:
        """Borrow exactly the saved result leaves, importing Torch only in a worker."""
        import torch

        from shadowspill.execution.results import TrainingResult

        if self.forward:
            if not isinstance(value, (tuple, list)):
                raise ValueError("worker forward results require grouped output pairs")
            groups = value
        else:
            if not isinstance(value, TrainingResult) or value.parameter_metrics:
                raise ValueError(
                    "worker training result differs from its prepared outputs"
                )
            if len(value.losses) != len(self.buffers) or len(value.metrics) != len(
                self.buffers
            ):
                raise ValueError("worker output count differs from its saved buffers")
            groups = tuple(zip(value.losses, value.metrics, strict=True))
        if len(groups) != len(self.buffers):
            raise ValueError("worker output count differs from its saved buffers")
        fields: dict[tuple[str, ...], torch.Tensor] = {}
        for index, (group, names, loss_present) in enumerate(
            zip(groups, self.names, self.losses, strict=True)
        ):
            if not isinstance(group, (tuple, list)) or len(group) != 2:
                raise ValueError("worker result requires a loss and values pair")
            loss, values = group
            if not isinstance(values, Mapping) or values.keys() != set(names):
                raise ValueError("worker returned a different named output set")
            if loss_present:
                fields["inputs", str(index), "loss"] = loss
            elif loss is not None:
                raise ValueError("worker returned an undeclared loss")
            for name in names:
                fields["inputs", str(index), "values", name] = values[name]
        for path, tensor in fields.items():
            if (
                not isinstance(tensor, torch.Tensor)
                or tuple(tensor.shape) != self.frame.shapes[path]
                or str(tensor.dtype) != "torch." + self.frame.dtypes[path]
                or tensor.layout != torch.strided
                or tensor.is_conj()
                or tensor.is_neg()
            ):
                raise ValueError("worker tensor differs from its saved result ABI")
        return MappingProxyType(fields)

    def write(
        self, value: Any, stream: BufferWriter, budget: BufferBudget, artifact_id: str
    ) -> ArtifactRef:
        from shadowspill.execution.tensor_copy import host_chunks

        fields = dict(self.fields(value))
        try:
            return self.frame.write_chunks(
                stream,
                lambda path: host_chunks(fields[path], chunk_bytes=FRAME_CHUNK_BYTES),
                budget,
                artifact_id,
            )
        finally:
            fields.clear()


def prepare_output_frames(
    prepared: PreparedPlans, rank: int
) -> Mapping[tuple[str, str], WorkerOutputFrame]:
    """Resolve every saved compute/optimizer output before starting a worker."""
    result: dict[tuple[str, str], WorkerOutputFrame] = {}
    for rows in prepared.families.values():
        for candidate in rows:
            worker = candidate.workers[rank]
            bindings = tuple(
                binding
                for (_, logical_rank), binding in sorted(
                    candidate.source.layout.bindings.items()
                )
                if logical_rank == rank
            )
            for name in worker.members():
                frame = WorkerOutputFrame(
                    prepared.plans[rank][name].program,
                    () if name == worker.optimizer else bindings,
                )
                key = candidate.id, name
                previous = result.setdefault(key, frame)
                if (
                    previous.frame.header != frame.frame.header
                    or previous.buffers != frame.buffers
                ):
                    raise ValueError(
                        "worker output identity has conflicting frame layouts"
                    )
    return MappingProxyType(result)
