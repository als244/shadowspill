"""Explicit search tracing survives the phase API without changing its answer."""

from dataclasses import replace

from shadowspill import MemoryBudgets, plan_program
from shadowspill.diagnostics.pressurefit_trace import PressureFitTrace, read_manifest

from ._example import communication, options, profiled


def test_cpu_plan_records_requested_search_and_preserves_selection(tmp_path):
    search = options()
    search = replace(search, generic=replace(search.generic, deterministic=True))
    arguments = dict(
        memory_budgets=MemoryBudgets(200, 64),
        communication_info=communication(),
    )
    original = plan_program(profiled(), search_options=search, **arguments)
    traced = plan_program(
        profiled(),
        search_options=replace(search, trace_directory=str(tmp_path)),
        **arguments,
    )
    assert traced.schedule == original.schedule
    assert traced.timeline == original.timeline
    directories = list(tmp_path.glob("*/manifest.json"))
    assert len(directories) == 1
    directory = directories[0].parent
    manifest = read_manifest(directory)
    assert manifest["status"] == "complete"
    assert (
        manifest["answer"]["candidate_id"] == traced.diagnostics.selected_candidate_id
    )
    assert manifest["answer"]["makespan_ns"] == traced.timeline.makespan_ns
    for candidate in manifest["candidates"]:
        PressureFitTrace.read(directory / candidate["file"]).verify()
    # Recording decisions never upgrades estimates into execution evidence.
    assert traced.simulation_only_reasons
