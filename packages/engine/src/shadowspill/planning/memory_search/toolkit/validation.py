"""Whether a search was handed something it can work on at all.

None of this is any one search's: the types are the planner's public
vocabulary, and the capacity relationship is arithmetic over what the
caller declared. A search calls it before doing anything else, and an
implementation living outside this repository calls the same function
rather than writing its own.
"""

from __future__ import annotations

from shadowspill.ir import (
    ResidencySpec,
    ShadowSpillProgram,
    shared_residency_footprint,
)
from shadowspill.planning.memory_search.admission import AdmissionFacts
from shadowspill.simulator import SimulationConfig


def validate_search_inputs(
    program: ShadowSpillProgram,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...],
    config: SimulationConfig,
    admission: AdmissionFacts | None,
) -> None:
    """Refuse search inputs of the wrong type, or whose capacities disagree.

    A search calls it before anything else. With ``admission``, the
    device's configured capacity less its shared residency must equal the
    facts' ``object_capacity_bytes``: the search plans against the first
    and the pool is admitted against the second, so a disagreement gives a
    plan that fits nothing.

    Parameters
    ----------
    program : ShadowSpillProgram
        The program.
    initial_residency : tuple of ResidencySpec
        Where each alias group is before the first task.
    final_residency : tuple of ResidencySpec
        Where each alias group must be after the last task.
    config : SimulationConfig
        The machine.
    admission : AdmissionFacts or None
        The execution pool's physical facts, or None to check types only.

    Raises
    ------
    TypeError
        If an argument has the wrong type.
    ValueError
        If the facts do not describe the program, or the capacities disagree.
    """

    if not isinstance(program, ShadowSpillProgram):
        raise TypeError("program must be a ShadowSpillProgram")
    if not isinstance(initial_residency, tuple):
        raise TypeError("initial_residency must be a tuple")
    if not isinstance(final_residency, tuple):
        raise TypeError("final_residency must be a tuple")
    if not isinstance(config, SimulationConfig):
        raise TypeError("config must be a SimulationConfig")
    if admission is None:
        return
    if not isinstance(admission, AdmissionFacts):
        raise TypeError("admission must be an AdmissionFacts")
    admission.validate(program)
    configured = {item.device_id: item for item in config.devices}
    shared = shared_residency_footprint(program)
    movable_capacity = (
        configured[admission.device_id].capacity_bytes
        - shared.for_device(admission.device_id)
        if admission.device_id in configured
        else None
    )
    if (
        admission.device_id not in configured
        or movable_capacity != admission.object_capacity_bytes
    ):
        raise ValueError(
            "feasibility capacity after shared residency must equal "
            "AdmissionFacts object capacity"
        )


__all__ = ["validate_search_inputs"]
