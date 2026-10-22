"""Grouping is experiment policy; preparation and search are library APIs."""

from pathlib import Path


def extend_search(request, settings):
    """Additional budgets within a prepared domain need only a CPU search."""
    directory = Path(request["directory"])
    if not (directory / "training.json").exists():
        return False
    from shadowspill.pytorch import search_prepared_steps
    from shadowspill.training import PreparedTraining

    prepared = PreparedTraining.load(directory)
    if not prepared.candidates:
        # Every candidate exhausted the preparation device. A larger request
        # may now permit profiling, so retry preparation instead of CPU search.
        return False
    budgets = [tuple(item) for item in request["budgets"]]
    if any(
        execution > program.problem.maximum_execution_budget_bytes
        or spill > program.problem.maximum_spill_budget_bytes
        for candidate in prepared.candidates.values()
        for program in candidate.programs
        for execution, spill in budgets
    ):
        return False
    report = search_prepared_steps(
        prepared.candidates,
        budgets=budgets,
        search_options=prepared.search.search_options,
        artifact_store=None
        if settings.artifact_store == "default"
        else settings.artifact_store,
        progress=lambda message: print(message, flush=True),
    )
    PreparedTraining(prepared.candidates, report).save(directory)
    return True
