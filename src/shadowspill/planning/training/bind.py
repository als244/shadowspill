"""Select an example traversal using only saved IR and call metadata."""

from dataclasses import replace

from shadowspill.artifacts._values import plain
from shadowspill.artifacts.program import Program
from shadowspill.ir import MemoryLocation, ResidencySpec
from shadowspill.step.ordering import StepDataOrdering

from .tasks import assemble_training


def bind_training(
    program: Program,
    ordering: StepDataOrdering | None = None,
    *,
    accumulate: bool = False,
) -> Program:
    templates = program.bindings.get("training_templates")
    if templates is None:
        if ordering is not None or accumulate:
            raise ValueError("this Program does not have training example ordering")
        return program
    default = StepDataOrdering.depth_first(len(templates))
    ordering = ordering or default
    if not isinstance(ordering, StepDataOrdering):
        raise TypeError("ordering must be StepDataOrdering")
    constraints = program.bindings.get("ordering_constraints", ())
    if constraints and ordering != default:
        raise ValueError("unsupported reordering: " + "; ".join(constraints))
    aliases = {item.object_id: item.alias_group_id for item in program.objects}
    gradients = tuple(
        item["gradient_object_id"] for item in program.bindings["gradients"]
    )
    sequence = assemble_training(
        templates,
        aliases,
        ordering,
        device_id=program.devices[0].device_id,
        accumulate=accumulate,
        gradients=gradients,
    )
    initial = {item.alias_group_id: item for item in program.initial_residency}
    for value in gradients:
        alias = aliases[value]
        if accumulate:
            initial[alias] = ResidencySpec(alias, MemoryLocation.SPILL)
        else:
            initial.pop(alias, None)
    return replace(
        program,
        tasks=tuple(sequence.tasks),
        initial_residency=tuple(initial.values()),
        task_alternative_groups=tuple(sequence.groups),
        bindings={
            **plain(program.bindings),
            "entrypoints": sequence.entrypoints,
            "transient_outputs": sequence.transient_outputs,
            "ordering": ordering.to_dict(),
            "accumulate_gradients": accumulate,
        },
    )
