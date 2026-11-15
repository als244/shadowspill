"""CPU-only base provenance; unchanged-storage proof stays in the checkpoint codec."""

from collections.abc import Sequence
from dataclasses import replace

from shadowspill.checkpoint.frozen import omitted_from_programs
from shadowspill.training.preparation.family_records import PreparedPlans
from shadowspill_client.protocol.frozen import FrozenBase
from shadowspill_client.protocol.requests import ArtifactRef
from shadowspill_client.protocol.training import ResolvedTraining


def validate_base(base: FrozenBase, configuration: ResolvedTraining) -> None:
    if (
        base.source != configuration.definition_sources["model"]
        or replace(configuration.model, initialize=base.model.initialize) != base.model
    ):
        raise ValueError("checkpoint frozen base differs from the resolved model")


def fresh_base(
    family: PreparedPlans, weights: Sequence[ArtifactRef]
) -> FrozenBase | None:
    """Record initialization only when at least one owned frozen tensor is omitted."""
    for plans in family.plans.values():
        primary = min(plans)
        if not any(
            item["parameter"] and not item["trainable"]
            for item in plans[primary].program.bindings["registrations"]
        ):
            continue
        if omitted_from_programs(
            plans[primary].program,
            (plan.program for name, plan in plans.items() if name != primary),
        ):
            model = family.inventory.layouts.setup.config.model
            initial = model.initialize
            if initial is None:
                raise ValueError(
                    "worker creation requires explicit weights or a checkpoint"
                )
            artifact = None
            if initial.kind == "artifact":
                artifact = next(
                    (ref for ref in weights if ref.id == initial.reference), None
                )
            return FrozenBase(
                model, family.inventory.layouts.setup.model.source, artifact
            )
    return None
