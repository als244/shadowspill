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
from shadowspill.simulator import SimulationConfig

from ...admission import AdmissionFacts


def validate_search_inputs(
    program: ShadowSpillProgram,
    initial_residency: tuple[ResidencySpec, ...],
    final_residency: tuple[ResidencySpec, ...],
    config: SimulationConfig,
    admission: AdmissionFacts | None,
) -> None:
    """Refuse malformed inputs, and a capacity that does not reconcile.

    The types first, so a mistake is named where it was made rather than
    several layers down as a wrong-shaped struct. Then the one relationship
    a caller can get wrong quietly: the capacity left after shared
    residency must be exactly the object capacity the `AdmissionFacts`
    declares, because the search plans against the first and the pool is
    admitted against the second. A disagreement there produces a plan that
    fits nothing.

    Raises `TypeError` for a wrong type and `ValueError` for a capacity
    that does not reconcile. Returns `None` when there is nothing to say.
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
