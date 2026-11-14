"""A configuration entry point; GPU work belongs to fresh phase workers."""

from shadowspill.training.preparation.phases import run_phases

from .configuration import configure


def run(recipe, config, *, phase="all"):
    """Run an importable recipe adapter through the shared training pipeline.

    The adapter supplies model/data setup and explicit build candidates. It
    does not launch phases, plan schedules or execute training loops. Named
    phases reload their saved choices; only planning may change memory budgets.
    Neither this entry point nor CPU planning imports a model or initializes CUDA.
    """
    root, options = configure(recipe, config, phase)
    run_phases(
        "shadowspill.training.preparation",
        root,
        phase,
        ranks=options["ranks"],
        preparation_ranks=options["preparation_ranks"],
    )
    return root
