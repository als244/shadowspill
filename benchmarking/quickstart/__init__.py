"""Generic search/run/plot entrypoint, with optional CLI text presets.

A factory receives the selected local device after runtime installation and
returns ordinary planning arguments. It need not subclass or register anything.
"""

from __future__ import annotations

from .cli import main as main


def run(
    factory,
    *,
    search_budget_gib,
    spill_gib,
    run_budget_gib=None,
    steps=5,
    output_dir=None,
    artifact_store=None,
    device="auto",
    plots=True,
    resolution_plans=True,
    timelines=True,
    profiling_options=None,
    control_group=None,
    host_headroom_gib=2,
    preparation_timeout=1800,
):
    """Benchmark a user experiment; factory(device=...) supplies its arguments.

    Required factory keys: model_factory (or an initialized model), objective,
    optimizer, candidates. Optional keys: initialize, hyperparams, plan_options,
    metadata, units_per_step, unit_label and context. candidates maps names to
    sequences of positional microbatch inputs. Every candidate represents the
    same caller-normalized update. For distributed use, supply a caller-owned
    Gloo control_group and return distributed=Distributed(...) (or a function
    of the fresh model returning it). Factories create their CUDA groups after
    Runtime installation and own their cleanup through context.
    """
    from pathlib import Path

    from shadowspill.pytorch.accelerator import resolve_device

    from .cli import execute
    from .options import _parser
    from .runner import Request

    arguments = _parser().parse_args([])
    arguments.factory = f"{factory.__module__}:{factory.__qualname__}"
    arguments.steps = steps
    arguments.plots = plots
    arguments.resolution_plans = resolution_plans
    arguments.timelines = timelines
    arguments.output_dir = None if output_dir is None else Path(output_dir)
    arguments.artifact_store = None if artifact_store is None else Path(artifact_store)
    arguments.search_budget_gib = list(search_budget_gib)
    arguments.run_budget_gib = list(
        search_budget_gib if run_budget_gib is None else run_budget_gib
    )
    arguments.spill_gib = spill_gib
    arguments.device = str(device)
    arguments.distributed = control_group is not None
    arguments.host_headroom_gib = host_headroom_gib
    arguments.preparation_timeout = preparation_timeout
    if steps < 1 or not search_budget_gib or spill_gib <= 0:
        raise ValueError(
            "positive steps, execution budgets and spill budget are required"
        )
    if any(v <= 0 for v in search_budget_gib) or any(
        v not in search_budget_gib for v in arguments.run_budget_gib
    ):
        raise ValueError("each positive run budget must be a search budget")
    if profiling_options is not None:
        from dataclasses import fields

        for item in fields(profiling_options):
            setattr(
                arguments, "profile_" + item.name, getattr(profiling_options, item.name)
            )
    request = Request(
        label=getattr(factory, "__name__", "experiment"),
        search_budgets=[int(v * (1 << 30)) for v in search_budget_gib],
        run_budgets=[int(v * (1 << 30)) for v in arguments.run_budget_gib],
        physical_capacity=int(max(search_budget_gib) * (1 << 30)),
        spill_budget=int(spill_gib * (1 << 30)),
        device=resolve_device(device),
    )
    return execute(arguments, request, factory, control_group=control_group)
