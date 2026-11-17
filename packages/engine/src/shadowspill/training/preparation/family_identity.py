"""Shared source and variant identities for planning and strict saved reuse."""

from shadowspill.artifacts._values import digest, plain
from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.step.ordering import StepDataOrdering

from .allocation_coverage import validate_allocation_coverage
from .family_records import WorkerPlans
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
        or source.metadata.get("external_headroom_bytes")
        != inventory.layouts.setup.config.preparation.external_headroom_bytes
    ):
        raise ValueError(
            "saved profiling policy/scenario differs from the prepared template"
        )
    if source.estimated_memory:
        raise ValueError("prepared operation family requires measured memory evidence")


def plan_identity(use: TemplateUse, ordering: StepDataOrdering | None) -> str:
    """A rank-local planning request, reusable across operation families."""
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


def execution_names(
    candidate: TemplateCandidate, ordering: StepDataOrdering | None
) -> WorkerPlans:
    """One invocation name for the complete set of rank-local Plan uses.

    Capture identities include logical rank and must remain distinct. Execution
    names instead identify the coordinated operation: peers must choose the same
    primary and compare the same named collective paths. Reuse is safe only when
    every rank's use matches, including count, traversal and capture identity.
    """

    def name(role: str) -> str | None:
        members = {
            str(rank): None if use is None else plan_identity(use, ordering)
            for rank, worker in sorted(candidate.workers.items())
            for use in (getattr(worker, role),)
        }
        if all(value is None for value in members.values()):
            return None
        if any(value is None for value in members.values()):
            raise ValueError("prepared workers disagree on operation dependencies")
        return digest(members)

    main = name("main")
    assert main is not None
    return WorkerPlans(main, name("optimizer"))


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
