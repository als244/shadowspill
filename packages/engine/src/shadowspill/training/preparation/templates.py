"""Count-independent capture inventory for the complete prepared layout family."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Literal

from shadowspill.artifacts.program import Program
from shadowspill.training.definitions.build import ModelBuild, ModelBuildContext
from shadowspill.training.definitions.samples import RepresentativePattern
from shadowspill.training.inputs.adapters import InputLayout
from shadowspill.training.inputs.shapes import PackingSchedule
from shadowspill.training.outputs import OutputLayout
from shadowspill_client import BuildOptions
from shadowspill_client._records import number, plain
from shadowspill_client.protocol.operations import OperationKey

from .binding import OperationBinding
from .layouts import LayoutCandidate, PreparedLayouts
from .operations import OperationContract

type ProgramKind = Literal["step", "forward_backward", "forward", "optim_step"]


@dataclass(frozen=True)
class CaptureTemplate:
    id: str
    binding: OperationBinding
    build: BuildOptions
    output_only: bool
    representative: RepresentativePattern
    model_build: ModelBuild
    # Canonical JSON keeps nested provenance immutable without retaining another
    # mutable tree beside the descriptions used by the build worker.
    identity: str

    @property
    def kind(self) -> ProgramKind:
        if self.output_only or self.binding.contract.key.operation == "forward":
            return "forward"
        if self.binding.contract.key.operation == "step":
            return "step"
        return "forward_backward"

    def signature(self) -> dict[str, Any]:
        return {"id": self.id, **json.loads(self.identity)}


@dataclass(frozen=True)
class TemplateUse:
    template: str
    kind: ProgramKind
    count: int


@dataclass(frozen=True)
class WorkerTemplates:
    main: TemplateUse
    masked: TemplateUse | None
    optimizer: TemplateUse | None


@dataclass(frozen=True)
class TemplateCandidate:
    layout: LayoutCandidate
    workers: Mapping[int, WorkerTemplates]


@dataclass(frozen=True)
class TemplateInventory:
    layouts: PreparedLayouts
    templates: Mapping[str, CaptureTemplate]
    families: Mapping[OperationKey, tuple[TemplateCandidate, ...]]


def capture_inventory(layouts: PreparedLayouts) -> TemplateInventory:
    """Deduplicate exact captures, never rank ownership or timing scenarios.

    A canonical output layout has one physical buffer per logical partition.
    Counts refer to that one saved form. Optimizer-backed backward uses the same
    step capture as complete updates; optimizer and persistent backward Programs
    are derived later. All-masked training keeps a separately captured forward.
    No model, tensor, policy object or GPU is constructed here.
    """
    setup = layouts.setup
    parallelism = setup.context.parallelism
    placement = parallelism.inputs
    assert placement is not None
    templates: dict[str, CaptureTemplate] = {}
    families: dict[OperationKey, tuple[TemplateCandidate, ...]] = {}
    patterns: dict[tuple[Any, ...], RepresentativePattern] = {}
    constructions: dict[tuple[Any, ...], ModelBuild] = {}

    def layout_key(layout: InputLayout) -> tuple[Any, ...]:
        return (
            layout.logical_rank,
            layout.buffer.max_datums,
            tuple(sorted(layout.buffer.fields.items())),
        )

    unique_layouts = {
        layout_key(binding.inputs.layout): binding.inputs.layout
        for candidates in layouts.families.values()
        for row in candidates
        for binding in row.bindings.values()
    }
    model_family = tuple(unique_layouts[key] for key in sorted(unique_layouts))

    def remember(
        binding: OperationBinding, row: LayoutCandidate, *, output_only: bool
    ) -> str:
        build = setup.config.preparation.cases[row.candidate.case].build
        physical = binding.inputs.layout.buffer
        pattern_key = (
            row.candidate.batch,
            physical.max_datums,
            tuple(sorted(physical.fields.items())),
        )
        if pattern_key not in patterns:
            describe = setup.preset.definition.representative
            if describe is None:
                raise ValueError(
                    "preparation preset must declare a representative pattern"
                )
            pattern = describe(setup.context, row.candidate.batch, physical)
            if not isinstance(pattern, RepresentativePattern):
                raise TypeError("preset must return a RepresentativePattern")
            pattern.describe(setup.context.schema, physical)
            patterns[pattern_key] = pattern
        pattern = patterns[pattern_key]
        construction_key = layout_key(binding.inputs.layout)
        if construction_key not in constructions:
            constructions[construction_key] = setup.model.describe_build(
                ModelBuildContext(binding.inputs.layout, model_family, parallelism)
            )
        model_build = constructions[construction_key]
        provenance = {
            "version": 1,
            "binding": binding.signature(),
            "build": build.to_dict(),
            "model_build": model_build.to_dict(),
            "allocation_invariants": {
                "model": dict(model_build.allocation_invariants),
                "objective": dict(setup.objective.definition.allocation_invariants),
                "optimizer": {}
                if setup.optimizer is None
                or output_only
                or binding.contract.key.operation == "forward"
                else dict(setup.optimizer.definition.allocation_invariants),
            },
            "policy_sources": {
                key: value.to_dict()
                for key, value in setup.config.policy_sources.items()
            },
            "parallelism": parallelism.to_dict(),
            "engine_source": setup.config.engine_source.to_dict(),
            "client_source": setup.config.client_source.to_dict(),
            "optimizer": None
            if setup.config.optimizer is None or output_only
            else setup.config.optimizer.to_dict(),
            "optimizer_source": None
            if setup.optimizer is None or output_only
            else setup.optimizer.source.to_dict(),
            "output_only": output_only,
            "scenario": {
                "preset": setup.config.preparation.preset,
                "source": setup.config.preparation.source.to_dict(),
                "typical": dict(setup.config.batches[row.candidate.batch].typical),
                "pattern": pattern.to_dict(),
                "measurement_values": (
                    "captured controls; other inputs use "
                    "core representative value policy"
                ),
            },
        }
        identity = json.dumps(plain(provenance), sort_keys=True, separators=(",", ":"))
        if len(identity.encode()) > 1 << 20:
            raise ValueError("capture template description exceeds 1 MiB")
        key = sha256(identity.encode()).hexdigest()
        templates.setdefault(
            key,
            CaptureTemplate(
                key, binding, build, output_only, pattern, model_build, identity
            ),
        )
        return key

    for key, candidates in layouts.families.items():
        family = []
        for row in candidates:
            schedule = row.candidate.schedule
            # Expansion currently repeats a single form within each partition.
            # Refuse an unsupported mixed traversal instead of silently capturing
            # its first buffer or changing collective participation.
            if any(
                not part or any(buffer != part[0] for buffer in part)
                for part in schedule.partitions
            ):
                raise ValueError("capture inventory requires uniform partition forms")
            canonical = PackingSchedule(
                "capture", [[part[0]] for part in schedule.partitions]
            )
            training = key.operation != "forward"
            capture_key = (
                OperationKey("step", key.outputs, key.mode)
                if training and setup.optimizer is not None
                else key
            )
            contract = OperationContract(capture_key, setup.objective)
            outputs = OutputLayout(
                setup.context.schema, contract.outputs, canonical, placement
            )
            workers = {}
            representative_capacities = []
            offset = 0
            for partition, part in enumerate(schedule.partitions):
                for rank in placement.partitions.groups[partition]:
                    binding = OperationBinding(
                        setup.model,
                        contract,
                        row.bindings[offset, rank].inputs,
                        outputs,
                        partition,
                    )
                    main = remember(binding, row, output_only=False)
                    masked = (
                        remember(binding, row, output_only=True) if training else None
                    )
                    workers[rank] = WorkerTemplates(
                        TemplateUse(main, _kind(key), len(part)),
                        None
                        if masked is None
                        else TemplateUse(masked, "forward", len(part)),
                        TemplateUse(main, "optim_step", 1)
                        if key.operation == "forward_backward"
                        and setup.optimizer is not None
                        else None,
                    )
                # One logical copy per partition, independent of cooperating
                # workers. Timing samples must obey all actual input limits as
                # well as physical geometry (including unscheduled limits).
                representative_capacities.append(
                    templates[main].representative.capacities(
                        setup.context.schema, part[0]
                    )
                )
                offset += len(part)
            for capacity in setup.context.schema.capacities:
                sizes = [values[capacity.name] for values in representative_capacities]
                total = max(sizes) if capacity.reduction == "max" else sum(sizes)
                if total > number(
                    setup.config.batches[row.candidate.batch].limits[capacity.name],
                    capacity.name,
                ):
                    raise ValueError(f"representative pattern exceeds {capacity.name}")
            family.append(TemplateCandidate(row, MappingProxyType(workers)))
        families[key] = tuple(family)
    return TemplateInventory(
        layouts, MappingProxyType(templates), MappingProxyType(families)
    )


def _kind(key: OperationKey) -> ProgramKind:
    if key.operation not in ("forward", "forward_backward", "step"):
        raise ValueError("optimizer dependencies are derived from a training template")
    return key.operation


def instantiate_template(
    template: CaptureTemplate, use: TemplateUse, program: Program
) -> Program:
    """Select/derive saved forms on CPU, without capture or kernel compilation."""
    from shadowspill.planning.training.backward import backward_program
    from shadowspill.planning.training.examples import instantiate_examples
    from shadowspill.planning.training.operations import optimizer_program

    validate_template(template, program)
    if use.template != template.id or type(use.count) is not int or use.count < 1:
        raise ValueError("saved capture differs from the requested template")
    if use.kind == "optim_step":
        if template.kind != "step" or use.count != 1:
            raise ValueError("optimizer requires one saved step template")
        return optimizer_program(program)
    if use.kind == "forward_backward" and template.kind == "step":
        program = backward_program(program)
    elif use.kind != template.kind:
        raise ValueError("saved template cannot supply the requested operation")
    # Even forward count one uses the grouped ABI expected by service staging and
    # output publication. Ordinary core forward Programs keep their original ABI.
    return instantiate_examples(program, [0] * use.count)


def validate_template(template: CaptureTemplate, program: Program) -> None:
    preparations = program.bindings.get("operator_preparation", {}).get("providers", ())
    if (
        program.bindings.get("kind") != template.kind
        or plain(program.bindings.get("preparation_template")) != template.signature()
        or plain(program.bindings.get("profiling_scenario"))
        != template.signature()["scenario"]
        or program.resources != template.model_build.resources
        or [
            {name: plain(value[name]) for name in ("factory", "options")}
            for value in preparations
        ]
        != [value.to_dict() for value in template.model_build.operator_preparation]
    ):
        raise ValueError("saved capture differs from the requested template")
