"""One captured tensor graph, its storage contract and structural identity."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field, replace
from typing import Literal

import torch
from torch.fx import GraphModule

from shadowspill.errors import CaptureError
from shadowspill.pytorch.accelerator import provider_version
from shadowspill.pytorch.capture.live_storage import live_storage_identity
from shadowspill.pytorch.capture.storage import (
    ExplicitMutation,
    TaskStorageContract,
    capture_task_storage_contract,
)
from shadowspill.pytorch.representations import (
    is_wrapper,
)

from .geometry import TaskInputProvenance, TensorGeometry
from .identity import (
    _artifact_digest,
    _canonical_graph,
    _operator_targets,
    _output_count,
    _representation_identity,
)
from .inputs import (
    _compact_mutations,
    _normalize_input_provenance,
    _specialize_static_inputs,
    _tensor_alias_groups,
    _tensor_arguments,
)


@dataclass(frozen=True, slots=True)
class GraphArtifact:
    """One explicit tensor graph and its structural contract identity."""

    kind: Literal["forward", "backward", "inference", "optimizer"]
    graph_module: GraphModule
    tensor_inputs: tuple[TensorGeometry, ...]
    argument_count: int
    output_count: int
    operator_targets: tuple[str, ...]
    tensor_argument_positions: tuple[int, ...]
    tensor_argument_alias_groups: tuple[int, ...]
    input_provenance: tuple[TaskInputProvenance, ...]
    storage_contract: TaskStorageContract
    storage_contract_capture_ns: int = field(compare=False)
    compatibility_digest: str
    example_arguments: tuple[object, ...] = field(repr=False, compare=False)
    #: Physical compiled arguments mapped to logical stage inputs.
    input_components: tuple[tuple[int, tuple[str, ...]], ...] = ()

    @classmethod
    def input_compatibility_digest(
        cls,
        *,
        graph_module: GraphModule,
        example_inputs: tuple[object, ...],
        explicit_mutations: tuple[ExplicitMutation, ...] = (),
        input_provenance: tuple[TaskInputProvenance, ...] | None = None,
    ) -> str:
        """Identify a graph/input contract without evaluating graph outputs."""

        original_inputs = example_inputs
        graph_module, example_inputs, tensor_positions = _specialize_static_inputs(
            graph_module, example_inputs
        )
        normalized = _normalize_input_provenance(
            graph_module,
            original_inputs,
            tensor_positions,
            input_provenance,
        )
        tensor_arguments = tuple(
            value for value in example_inputs if isinstance(value, torch.Tensor)
        )
        alias_group_by_storage: dict[int, int] = {}
        identity: dict[str, object] = {
            "graph": _canonical_graph(graph_module),
            "inputs": [
                TensorGeometry.from_tensor(value).identity()
                for value in tensor_arguments
            ],
            "tensor_argument_positions": tensor_positions,
            # What each input *is* -- a parameter, an activation, a
            # constant -- decides how it may be given a value: an
            # activation may be invented, a parameter may not. Two graphs
            # that are the same shape over inputs of different kinds are
            # therefore not the same task, and an artifact captured for one
            # does not fit the other. Leaving this out let a stored graph
            # pair be handed to a stage it was not captured for, which was
            # caught only where the roles disagreed loudly enough to refuse.
            "input_roles": [item.role.value for item in normalized],
            "tensor_argument_alias_groups": [
                alias_group_by_storage.setdefault(
                    id(value) if is_wrapper(value) else live_storage_identity(value),
                    len(alias_group_by_storage),
                )
                for value in tensor_arguments
            ],
            "mutations": [
                {
                    "input_position": item.input_position,
                    "output_leaf_index": item.output_leaf_index,
                    "target": item.target,
                }
                for item in explicit_mutations
            ],
            "torch": torch.__version__,
            "provider": provider_version(),
        }
        # Every graph pair uses the logical-to-physical input contract, even
        # when each logical tensor has only one ordinary storage component.
        component_aliases: dict[int, int] = {}
        identity["representations"] = [
            _representation_identity(value, component_aliases)
            for value in tensor_arguments
        ]
        encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    @classmethod
    def capture(
        cls,
        *,
        kind: Literal["forward", "backward", "inference", "optimizer"],
        graph_module: GraphModule,
        example_inputs: tuple[object, ...],
        explicit_mutations: tuple[ExplicitMutation, ...] = (),
        input_provenance: tuple[TaskInputProvenance, ...] | None = None,
    ) -> GraphArtifact:
        original_inputs = example_inputs
        graph_module, example_inputs, tensor_positions = _specialize_static_inputs(
            graph_module, example_inputs
        )
        provenance = _normalize_input_provenance(
            graph_module,
            original_inputs,
            tensor_positions,
            input_provenance,
        )
        tensor_arguments = _tensor_arguments(example_inputs)
        tensor_inputs = tuple(
            TensorGeometry.from_tensor(value) for value in tensor_arguments
        )
        tensor_alias_groups = _tensor_alias_groups(tensor_arguments)
        operators = _operator_targets(graph_module)
        output_count = _output_count(graph_module)
        storage_contract, contract_capture_ns = _capture_artifact_storage(
            graph_module,
            example_inputs,
            _compact_mutations(explicit_mutations, tensor_positions),
        )
        digest = _artifact_digest(
            kind,
            graph_module,
            tensor_inputs,
            len(example_inputs),
            tensor_positions,
            tensor_alias_groups,
            output_count,
            operators,
            storage_contract,
        )
        return cls(
            kind=kind,
            graph_module=graph_module,
            tensor_inputs=tensor_inputs,
            argument_count=len(example_inputs),
            output_count=output_count,
            operator_targets=operators,
            tensor_argument_positions=tensor_positions,
            tensor_argument_alias_groups=tensor_alias_groups,
            input_provenance=provenance,
            storage_contract=storage_contract,
            storage_contract_capture_ns=contract_capture_ns,
            compatibility_digest=digest,
            example_arguments=example_inputs,
        )

    def rebind_examples(
        self,
        example_arguments: tuple[object, ...],
        *,
        input_provenance: tuple[TaskInputProvenance, ...] | None = None,
    ) -> GraphArtifact:
        """Bind equivalent live tensors without re-extracting graph semantics.

        Structural AOT cache hits reuse the exact same FX graph and storage
        contract.  Only their occurrence-specific tensor storages differ.  The
        complete tensor geometry and input-alias contract are sufficient to prove
        that the cached contract still applies; re-running the graph merely to
        rediscover that immutable contract is both redundant and expensive.
        """

        if len(example_arguments) != self.argument_count or any(
            not isinstance(value, torch.Tensor) for value in example_arguments
        ):
            raise CaptureError(
                "rebound graph arguments differ from the tensor contract"
            )
        tensors = tuple(
            value for value in example_arguments if isinstance(value, torch.Tensor)
        )
        geometry = tuple(TensorGeometry.from_tensor(value) for value in tensors)
        if geometry != self.tensor_inputs:
            raise CaptureError(
                "rebound graph tensor geometry differs from its contract"
            )
        alias_group_by_storage: dict[int, int] = {}
        alias_groups = tuple(
            alias_group_by_storage.setdefault(
                live_storage_identity(value), len(alias_group_by_storage)
            )
            for value in tensors
        )
        if alias_groups != self.tensor_argument_alias_groups:
            raise CaptureError("rebound graph input aliases differ from its contract")
        provenance = self.input_provenance
        if input_provenance is not None:
            if len(input_provenance) != len(provenance):
                raise CaptureError("rebound graph input provenance arity changed")
            if tuple(item.role for item in input_provenance) != tuple(
                item.role for item in provenance
            ):
                raise CaptureError("rebound graph input provenance roles changed")
            provenance = input_provenance
        return replace(
            self,
            example_arguments=example_arguments,
            input_provenance=provenance,
        )


def _capture_artifact_storage(
    graph_module: GraphModule,
    example_inputs: tuple[object, ...],
    mutations: tuple[ExplicitMutation, ...],
) -> tuple[TaskStorageContract, int]:
    started = time.perf_counter_ns()
    contract = capture_task_storage_contract(
        graph_module,
        example_inputs,
        explicit_mutations=mutations,
    )
    return contract, time.perf_counter_ns() - started
