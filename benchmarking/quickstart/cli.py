"""Command-line composition of generic benchmarking and optional text presets."""

from __future__ import annotations

import functools
import importlib
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

from shadowspill.pytorch.accelerator import resolve_device

from .options import parse_arguments
from .reporting import gib, rule
from .runner import Request, Tour, open_runtime, plan_budgets, print_closing
from .storage import Ledger, console_log, default_run_root, prepare_run_root

GIB = 1 << 30


def resolve_request(parser, arguments):
    device = resolve_device(arguments.device)
    defaults = {}
    if arguments.factory is not None:
        module, separator, name = arguments.factory.partition(":")
        if not separator or not module or not name:
            parser.error("--factory must use module:function")
        factory = getattr(importlib.import_module(module), name)
        setup = functools.partial(factory, **arguments.factory_args)
        label = name
        manual = None
    else:
        from workloads.recipes.text.quickstart import recipe

        setup, defaults, manual = recipe(parser, arguments)
        label = arguments.model
    requested = arguments.search_budget_gib
    run = arguments.run_budget_gib
    if not requested and not run:
        run = [defaults.get("execution_gib", 8)]
    requested = sorted(requested or run)
    run = run or []
    if any(value not in requested for value in run):
        parser.error("each run budget must also be a search budget")
    spill = arguments.spill_gib or defaults.get("spill_gib", 16)
    if spill <= 0 or any(value <= 0 for value in requested):
        parser.error("execution and spill budgets must be positive")
    remote = None
    if arguments.remote_spill is not None:
        host, colon, port = arguments.remote_spill.rpartition(":")
        if not host or not colon or not port.isdigit():
            parser.error("--remote-spill must be HOST:PORT")
        remote = (host, int(port))
    return Request(
        label=label,
        search_budgets=[int(value * GIB) for value in requested],
        run_budgets=[int(value * GIB) for value in run],
        physical_capacity=int(max(requested) * GIB),
        spill_budget=int(spill * GIB),
        device=device,
        manual=manual,
        remote_spill=remote,
    ), setup


def execute(arguments, request, setup, *, control_group=None):
    control = None
    rank = None
    if control_group is not None:
        import torch.distributed as dist

        from shadowspill.pytorch.distributed._control import Control

        control = Control(
            control_group, namespace="quickstart", timeout=arguments.preparation_timeout
        )
        rank = dist.get_rank()
        control.agree(
            "request",
            {
                "label": request.label,
                "search_budgets": request.search_budgets,
                "run_budgets": request.run_budgets,
                "steps": arguments.steps,
                "manual": request.manual,
            },
        )
        proposed = str(default_run_root(arguments, request))
        arguments.output_dir = Path(control.exchange("output_root", proposed)[0])
        paths = control.run(
            "output_paths", lambda: prepare_run_root(arguments, request, rank=rank)
        )
        available = control.exchange("output_paths_ready", paths is not None)
        if not all(available):
            return 1
    else:
        paths = prepare_run_root(arguments, request)
    if paths is None:
        return 1

    def perform():
        with console_log(paths.root):
            ledger = Ledger()
            with open_runtime(
                request,
                ledger,
                control_group=control_group,
                host_headroom_gib=arguments.host_headroom_gib,
                preparation_timeout=arguments.preparation_timeout,
            ) as runtime:
                # CUDA model resources are created after combined host admission.
                experiment = setup(device=request.device)
                budgets = plan_budgets(runtime, request)
                print(rule(f"ShadowSpill quickstart: {request.label}"))
                print(f"  device: {request.device}; spill: {gib(request.spill_budget)}")
                if rank is not None:
                    print(f"  rank: {rank}; reports and stores are rank-local")
                print(f"  search budgets: {budgets.asked(budgets.requested_search)}")
                print(
                    f"  run budgets: {budgets.asked(budgets.requested_run) or 'none'}"
                )
                context = experiment.get("context", nullcontext)
                with context():
                    tour = Tour(
                        arguments, request, paths, budgets, runtime, ledger, experiment
                    )
                    try:
                        tour.search()
                        tour.plot_search()
                        entries = tour.run()
                        tour.plot_run(entries)
                    finally:
                        tour.close()
            print_closing(ledger, request)
        return 0

    # This is one setup/completion boundary around the tour. No control barriers
    # are added between measured runtime tasks or updates.
    return perform() if control is None else control.run("tour", perform)


def main():
    parser, arguments = parse_arguments()
    if not arguments.distributed:
        request, setup = resolve_request(parser, arguments)
        return execute(arguments, request, setup)

    import torch.distributed as dist

    dist.init_process_group(
        "gloo", timeout=timedelta(seconds=arguments.preparation_timeout)
    )
    try:
        request, setup = resolve_request(parser, arguments)
        return execute(arguments, request, setup, control_group=dist.group.WORLD)
    finally:
        dist.destroy_process_group()
