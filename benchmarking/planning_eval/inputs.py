"""Adapt measured phase artifacts at the CPU planning boundary."""

from dataclasses import replace

from shadowspill.artifacts.profiled_program import ProfiledProgram
from shadowspill.communication.info import CommunicationInfo
from shadowspill.planner.program_inputs import ShadowSpillPlanningProblem
from shadowspill.planning.admission.program import program_admission
from shadowspill.planning.budgets import MemoryBudgets
from shadowspill.planning.costs import simulation_inputs
from shadowspill.planning.training.bind import bind_training

from .config import FrontierConfig


def planning_problem(
    source: ProfiledProgram, config: FrontierConfig
) -> ShadowSpillPlanningProblem:
    """Bind saved compute/memory evidence to this sweep's explicit limits."""
    if (
        source.program.parallelism is not None
        and source.program.parallelism.world_size != 1
    ):
        raise ValueError("the planning corpus currently collects single-rank programs")
    profiled = replace(source, program=bind_training(source.program))
    rates = config.transfer_bandwidths
    communication = CommunicationInfo(
        fetch_bytes_per_second=rates.fetch_solo_bytes_per_second,
        evict_bytes_per_second=rates.evict_solo_bytes_per_second,
        fetch_concurrent_bytes_per_second=rates.fetch_concurrent_bytes_per_second,
        evict_concurrent_bytes_per_second=rates.evict_concurrent_bytes_per_second,
        metadata={"source": rates.provenance},
    )
    budgets = MemoryBudgets(
        max(size for grid in config.grids for size in grid.execution_budgets),
        max(size for grid in config.grids for size in grid.spill_budgets),
    )
    costed, simulation = simulation_inputs(profiled, budgets, communication)
    facts = program_admission(
        profiled, costed, simulation.devices[0].capacity_bytes, communication
    )
    if facts is None:
        raise ValueError("planning evaluation requires measured allocation evidence")
    return ShadowSpillPlanningProblem(
        role="step",
        program=costed,
        initial_residency=profiled.program.initial_residency,
        final_residency=profiled.program.final_residency,
        simulation_config=simulation,
        admission_facts=facts,
        source_execution_budget_bytes=budgets.execution_bytes,
        maximum_execution_budget_bytes=budgets.execution_bytes,
        maximum_spill_budget_bytes=budgets.spill_bytes,
        fixed_execution_bytes=profiled.fixed_device_bytes,
        object_reserve_bytes=0,
        dynamic_scratch_reserve_bytes=0,
    )


def program_digest(source: ProfiledProgram) -> str:
    """Verify a selected plan against the source's default ordered compute graph."""
    return replace(source, program=bind_training(source.program))._costed_ir().digest
