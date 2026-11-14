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

from shadowspill.ir import ShadowSpillProgram, TaskAlternativeChoice
from shadowspill.planner.admission import AdmissionFacts
from shadowspill.planner.admission.indexing import (
    IndexedAdmissionFacts,
    index_admission_facts,
)
from shadowspill.simulator import SimulationConfig
from shadowspill.simulator.indexing import (
    IndexedSimulationTemplate,
    index_simulation_template,
)


@dataclass(frozen=True, slots=True)
class AdmissionSetup:
    """The schedule-invariant half of measuring one resolved program."""

    template: IndexedSimulationTemplate
    indexed_facts: IndexedAdmissionFacts
    #: Per selected task, in template order, the allocation ordinal of each of
    #: its allocation steps, in the order the indexed facts flatten them. A
    #: certificate names a step's lease by its task and ordinal.
    allocation_ordinals: tuple[tuple[int, ...], ...]
    action_trigger_tasks: tuple[int, ...] = ()

    @property
    def task_ids(self) -> tuple[str, ...]:
        return self.template.task_ids

    @property
    def alias_ids(self) -> tuple[str, ...]:
        return self.template.alias_ids


def build_admission_setup(
    program: ShadowSpillProgram,
    selections: tuple[TaskAlternativeChoice, ...],
    config: SimulationConfig,
    facts: AdmissionFacts,
    *,
    template: IndexedSimulationTemplate | None = None,
) -> AdmissionSetup:
    """Compile the parts of admission a schedule cannot change.

    `template` is the resolved program already indexed, when the caller holds
    it: the one a plan was simulated on, say.
    """

    facts.validate(program)
    if template is None:
        template = index_simulation_template(program, selections, config)
    indexed_facts = index_admission_facts(facts, template)
    contracts = {item.task_id: item for item in facts.tasks}
    return AdmissionSetup(
        template=template,
        indexed_facts=indexed_facts,
        allocation_ordinals=tuple(
            tuple(
                step.allocation_ordinal for step in contracts[task_id].allocation_steps
            )
            for task_id in template.task_ids
        ),
    )


__all__ = ["AdmissionSetup", "build_admission_setup"]
