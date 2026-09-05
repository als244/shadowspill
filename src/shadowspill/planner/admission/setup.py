"""What every measurement of one resolved program shares.

A resolved program fixes which tasks execute; only the schedule varies below
it. Everything a measurement needs that a schedule does not change is prepared
here once and reused across every candidate policy and every capacity target
beneath it.

What each field holds, and why the cadence is per resolved program rather
than per plan or per candidate, is specified in
docs/architecture/admission-leases.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from shadowspill.ir import Program, TaskAlternativeChoice
from shadowspill.planner.admission import AdmissionFacts
from shadowspill.planner.admission.admission_replay import AdmissionReplayPurpose
from shadowspill.planner.admission.indexed import (
    IndexedAdmissionFacts,
    index_admission_facts,
)
from shadowspill.simulator import SimulationConfig
from shadowspill.simulator.indexed import (
    IndexedSimulationTemplate,
    index_simulation_template,
)


@dataclass(frozen=True, slots=True)
class AllocationStep:
    """One task allocation step, in the order the facts flattens them.

    `slot` is the lease it uses: a step that reallocates an earlier ordinal's
    slot shares its lease and emits no operation of its own.
    """

    task_id: str
    ordinal: int
    slot: int
    allocates: bool
    purpose: AdmissionReplayPurpose
    alias_group_id: str | None


@dataclass(frozen=True, slots=True)
class AdmissionSetup:
    """The schedule-invariant half of measuring one resolved program."""

    template: IndexedSimulationTemplate
    indexed_facts: IndexedAdmissionFacts
    allocation_steps: tuple[AllocationStep, ...]
    storage_handoffs: tuple[tuple[str, str], ...]
    action_trigger_tasks: tuple[int, ...] = ()

    @property
    def task_ids(self) -> tuple[str, ...]:
        return self.template.task_ids

    @property
    def alias_ids(self) -> tuple[str, ...]:
        return self.template.alias_ids


def build_admission_setup(
    program: Program,
    selections: tuple[TaskAlternativeChoice, ...],
    config: SimulationConfig,
    facts: AdmissionFacts,
) -> AdmissionSetup:
    """Compile the parts of admission a schedule cannot change."""

    facts.validate(program)
    template = index_simulation_template(program, selections, config)
    contracts = {item.task_id: item for item in facts.tasks}
    sizes = {item.alias_group_id: item.size_bytes for item in program.alias_groups}

    steps: list[AllocationStep] = []
    handoffs: list[tuple[str, str]] = []
    next_slot = 0
    for task in program.selected_tasks(selections):
        contract = contracts.get(task.task_id)
        if contract is None:
            continue
        replacements = frozenset(contract.replacement_aliases)
        slot_by_ordinal: dict[int, int] = {}
        for step in contract.allocation_steps:
            allocates = step.kind.value == "allocate"
            if allocates:
                if step.reuses_allocation_ordinal is None:
                    slot = next_slot
                    next_slot += 1
                else:
                    slot = slot_by_ordinal[step.reuses_allocation_ordinal]
                slot_by_ordinal[step.allocation_ordinal] = slot
            else:
                slot = slot_by_ordinal[step.allocation_ordinal]
            alias = step.output_alias_group_id
            steps.append(
                AllocationStep(
                    task_id=task.task_id,
                    ordinal=step.allocation_ordinal,
                    slot=slot,
                    allocates=allocates,
                    purpose=_purpose(alias, replacements),
                    alias_group_id=alias,
                )
            )
        handoffs.extend(
            (item.source_alias_group_id, item.destination_alias_group_id)
            for item in contract.storage_handoffs
            # A zero-byte destination is not a handoff; the source keeps its
            # lease.
            if sizes.get(item.destination_alias_group_id, 0) != 0
        )

    return AdmissionSetup(
        template=template,
        indexed_facts=index_admission_facts(facts, template),
        allocation_steps=tuple(steps),
        storage_handoffs=tuple(handoffs),
    )


def _purpose(
    alias_group_id: str | None,
    replacements: frozenset[str],
) -> AdmissionReplayPurpose:
    """Anonymous scratch, a fresh output, or the generation one supersedes."""

    if alias_group_id is None:
        return AdmissionReplayPurpose.TASK_WORKSPACE
    if alias_group_id in replacements:
        return AdmissionReplayPurpose.MUTATION_REPLACEMENT
    return AdmissionReplayPurpose.TASK_OUTPUT


__all__ = ["AdmissionSetup", "AllocationStep", "build_admission_setup"]
