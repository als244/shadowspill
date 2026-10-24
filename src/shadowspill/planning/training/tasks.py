"""Bind saved stage forms to IR dependencies and contribution destinations."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from shadowspill.ir import (
    MutationSpec,
    ResourceKind,
    ResourceSpec,
    TaskAlternativeGroup,
    TaskAlternativeOption,
    TaskSpec,
)
from shadowspill.step.ordering import StepDataOrdering

from .order import task_id, traversal


def _unique(values: Any) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


@dataclass
class TrainingSequence:
    tasks: list[TaskSpec] = field(default_factory=list)
    entrypoints: list[dict[str, Any]] = field(default_factory=list)
    groups: list[TaskAlternativeGroup] = field(default_factory=list)
    transient_outputs: dict[str, tuple[int, ...]] = field(default_factory=dict)


class _Assembler:
    def __init__(
        self,
        templates: Sequence[Any],
        aliases: Mapping[str, str],
        ordering: StepDataOrdering,
        *,
        device_id: str,
        accumulate: bool,
        gradients: Sequence[str],
    ):
        self.aliases = aliases
        self.resource = ResourceSpec(device_id, ResourceKind.COMPUTE)
        self.result = TrainingSequence(
            tasks=[
                TaskSpec(
                    "invocation_start",
                    ResourceSpec(device_id, ResourceKind.CONTROL),
                    "invocation_start",
                    phase="control",
                    requires_entrypoint=False,
                )
            ]
        )
        self.writers: dict[str, tuple[str, ...]] = {
            value: ("invocation_start",) for value in gradients if accumulate
        }
        self.producers: dict[str, tuple[str, ...]] = {}
        self.forms = [
            [
                tuple(
                    form
                    for form in stage
                    if form["accumulates"]
                    == (
                        accumulate
                        or not ordering.creates(example, index, stage_count=len(stages))
                    )
                )
                for index, stage in enumerate(stages)
            ]
            for example, stages in enumerate(templates)
        ]
        if any(not forms for stages in self.forms for forms in stages):
            raise ValueError("Program is missing a required gradient form")

    def ids(self, phase: str, example: int, stage: int) -> tuple[str, ...]:
        if not 0 <= stage < len(self.forms[example]):
            return ()
        return tuple(
            task_id(phase, example, stage, f["variant"])
            for f in self.forms[example][stage]
        )

    def emit(self, phase: str, example: int, stage: int) -> None:
        forms = self.forms[example][stage]
        writers: dict[str, list[str]] = {}
        producers: dict[str, list[str]] = {}
        for form in forms:
            spec = form[phase]
            identifier = task_id(phase, example, stage, form["variant"])
            inputs = _unique(slot["object_id"] for slot in spec["inputs"])
            if phase == "forward":
                outputs = _unique(slot["object_id"] for slot in spec["outputs"])
                mutations = tuple(spec["mutations"])
                dependencies = self.ids(phase, example, stage - 1)
                transient = tuple(spec["replacements"])
            else:
                destinations = _unique(s["object_id"] for s in spec["contributions"])
                mutations = tuple(
                    value for value in destinations if value in self.writers
                )
                outputs = tuple(
                    value for value in destinations if value not in mutations
                )
                inputs = _unique((*inputs, *mutations))
                dependencies = (
                    task_id("forward", example, stage, form["variant"]),
                    *self.ids("backward", example, stage + 1),
                )
                transient = tuple(
                    dict.fromkeys(
                        (
                            *spec["replacements"],
                            *spec["discarded"],
                            *(
                                s["leaf_index"]
                                for s in spec["contributions"]
                                if s["object_id"] in mutations
                            ),
                        )
                    )
                )
            input_aliases = {self.aliases[value] for value in inputs}
            outputs = tuple(
                value for value in outputs if self.aliases[value] not in input_aliases
            )
            dependencies = _unique(
                (
                    "invocation_start",
                    *dependencies,
                    *(
                        writer
                        for value in (*inputs, *mutations)
                        for writer in (
                            *self.producers.get(value, ()),
                            *self.writers.get(value, ()),
                        )
                    ),
                )
            )
            self.result.tasks.append(
                TaskSpec(
                    identifier,
                    self.resource,
                    spec["profile_key"],
                    dependencies=dependencies,
                    inputs=inputs,
                    outputs=outputs,
                    mutations=tuple(MutationSpec(value) for value in mutations),
                    phase=phase,
                )
            )
            self.result.transient_outputs[identifier] = transient
            self.result.entrypoints.append(
                _entrypoint(identifier, phase, example, stage, form)
            )
            for value in (*outputs, *mutations):
                writers.setdefault(value, []).append(identifier)
            for value in outputs:
                producers.setdefault(value, []).append(identifier)
        self.writers.update({value: tuple(ids) for value, ids in writers.items()})
        self.producers.update({value: tuple(ids) for value, ids in producers.items()})

    def alternatives(self) -> None:
        for example, stages in enumerate(self.forms):
            for stage, forms in enumerate(stages):
                self.result.groups.append(
                    TaskAlternativeGroup(
                        f"recompute_{example:04d}_{stage:04d}",
                        tuple(
                            TaskAlternativeOption(
                                form["variant"],
                                (
                                    task_id("forward", example, stage, form["variant"]),
                                    task_id(
                                        "backward", example, stage, form["variant"]
                                    ),
                                ),
                                tuple(form["saved_aliases"]),
                            )
                            for form in forms
                        ),
                    )
                )


def _entrypoint(
    identifier: str,
    phase: str,
    example: int,
    stage: int,
    form: Mapping[str, Any],
) -> dict[str, Any]:
    spec = form[phase]
    return {
        "task_id": identifier,
        "input_slots": spec["inputs"],
        "output_slots": spec.get("outputs", ()),
        "replacement_output_leaves": spec.get("replacements", ()),
        "storage_handoffs": spec["handoffs"],
        "options": {
            "phase": phase,
            "stage_index": stage,
            "repetition": example,
            "variant": form["variant"],
            "public_output_count": len(spec.get("public_outputs", ())),
            "public_output_leaves": spec.get("public_outputs", ()),
            "contribution_slots": spec.get("contributions", ()),
        },
    }


def assemble_training(
    templates: Sequence[Any],
    aliases: Mapping[str, str],
    ordering: StepDataOrdering,
    *,
    device_id: str = "cuda_0",
    accumulate: bool = False,
    gradients: Sequence[str] = (),
) -> TrainingSequence:
    if ordering.microbatches != len(templates):
        raise ValueError("ordering must cover exactly num_examples")
    if not templates or len({len(stages) for stages in templates}) != 1:
        raise ValueError("examples must have the same stage structure")
    assembler = _Assembler(
        templates,
        aliases,
        ordering,
        device_id=device_id,
        accumulate=accumulate,
        gradients=gradients,
    )
    for phase, example, stage in traversal(ordering, len(templates[0])):
        assembler.emit(phase, example, stage)
    assembler.alternatives()
    return assembler.result
