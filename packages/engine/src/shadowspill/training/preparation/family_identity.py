"""Shared source and variant identities for planning and strict saved reuse."""

from shadowspill.artifacts._values import digest, plain
from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.step.ordering import StepDataOrdering

from .allocation_coverage import validate_allocation_coverage
from .templates import (
    TemplateCandidate,
    TemplateInventory,
    TemplateUse,
    validate_template,
)


def validate_profile(
    inventory: TemplateInventory, key: str, source: ProfiledProgram
) -> None:
    template = inventory.templates[key]
    validate_template(template, source.program)
    validate_allocation_coverage(template, source.program)
    if (
        source.metadata.get("preparation_template") != key
        or plain(source.metadata.get("profiling_scenario"))
        != template.signature()["scenario"]
        or plain(source.metadata.get("profiling_options"))
        != inventory.layouts.setup.config.preparation.profiling.to_dict()
    ):
        raise ValueError(
            "saved profiling policy/scenario differs from the prepared template"
        )
    if source.estimated_memory:
        raise ValueError("prepared operation family requires measured memory evidence")


def plan_identity(use: TemplateUse, ordering: StepDataOrdering | None) -> str:
    return digest(
        {
            "template": use.template,
            "kind": use.kind,
            "count": use.count,
            "ordering": None
            if ordering is None or use.kind not in ("step", "forward_backward")
            else ordering.label,
        }
    )


def candidate_identity(
    candidate: TemplateCandidate, ordering: StepDataOrdering | None
) -> str:
    shape = candidate.layout.candidate
    return digest(
        {
            "operation": candidate.layout.operation.to_dict(),
            "batch": shape.batch,
            "case": shape.case,
            "count": shape.count,
            "ordering": None if ordering is None else ordering.label,
        }
    )
