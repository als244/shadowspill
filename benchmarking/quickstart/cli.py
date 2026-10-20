"""Command-line composition of generic benchmarking and optional text presets."""

from __future__ import annotations

import functools
import importlib
import inspect
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

from shadowspill.pytorch.accelerator import resolve_device
from workloads.models import catalog

from .options import graph_pair_variants, parse_arguments
from .reporting import gib, rule
from .runner import Request, Tour, open_runtime, plan_budgets, print_closing
from .storage import Ledger, console_log, default_run_root, prepare_run_root

GIB = 1 << 30


def _factory_arguments(parser, factory, arguments):
    """Factory arguments, including explicitly selected LoRA and sharding flags.

    A factory decides its own trainability, so --lora and its targeting flags
    stay with the text presets. Rank, alpha and factor dtype are the knobs a
    LoRA recipe exposes under the same names whatever the model, so those
    three reach a factory that takes them -- and one that does not is refused
    rather than left to ignore a setting the request records. Explicit transfer
    sharding flags follow the same contract; the factory owns group construction.
    """

    given = {
        name: value
        for name, value in (
            ("lora_rank", arguments.lora_rank),
            ("lora_alpha", arguments.lora_alpha),
            ("lora_dtype", arguments.lora_dtype),
            ("transfer_shard_size", arguments.transfer_shard_size),
            (
                "shard_gradients",
                arguments.shard_gradients
                if arguments.transfer_shard_size is not None
                else None,
            ),
        )
        if value is not None
    }
    if not given:
        return dict(arguments.factory_args)
    parameters = inspect.signature(factory).parameters
    variadic = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    missing = sorted(name for name in given if name not in parameters)
    if missing and not variadic:
        parser.error(
            f"{arguments.factory} does not take {', '.join(missing)}; quickstart"
            " flags reach a factory only under those names"
        )
    twice = sorted(set(given) & set(arguments.factory_args))
    if twice:
        parser.error(f"{', '.join(twice)} given both as a flag and in --factory-args")
    return {**arguments.factory_args, **given}


def resolve_request(parser, arguments):
    device = resolve_device(arguments.device)
    defaults = {}
    factory_path = arguments.factory
    bound: dict = {}
    if arguments.model is not None:
        entry = catalog.entry(arguments.model)
        if entry.factory is not None:
            factory_path = entry.factory
            bound = dict(entry.factory_arguments)
    if factory_path is not None:
        module, separator, name = factory_path.partition(":")
        if not separator or not module or not name:
            parser.error("--factory must use module:function")
        factory = getattr(importlib.import_module(module), name)
        given = _factory_arguments(parser, factory, arguments)
        fixed = sorted(set(bound) & set(given))
        if fixed:
            parser.error(f"{arguments.model} fixes {', '.join(fixed)} for its factory")
        setup = functools.partial(factory, **bound, **given)
        label = name if arguments.model is None else arguments.model
        manual = None
    else:
        from workloads.recipes.text.quickstart import recipe

        setup, defaults, manual = recipe(parser, arguments)
        label = f"{arguments.model}_{arguments.implementation or 'mlops'}" + (
            "_lora" if arguments.lora else ""
        )
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
    pool = None
    if arguments.remote_spill is not None:
        host, colon, port = arguments.remote_spill.rpartition(":")
        if not host or not colon or not port.isdigit():
            parser.error("--remote-spill must be HOST:PORT")
        from shadowspill.network import remote

        pool = remote(capacity=int(spill * GIB), host=host, port=int(port))
    elif arguments.ssd_spill is not None:
        from shadowspill.ssd import ssd

        try:
            pool = ssd(
                capacity=int(spill * GIB),
                directory=arguments.ssd_spill,
                staging_bytes=arguments.ssd_staging_mib << 20,
                chunk_bytes=arguments.ssd_chunk_mib << 20,
                queue_depth=arguments.ssd_queue_depth,
                image=arguments.ssd_image,
            )
        except (TypeError, ValueError) as error:
            parser.error(str(error))
    image = None
    if arguments.ssd_image is not None:
        from shadowspill.pytorch.state.image import PoolImage

        try:
            image = PoolImage.read(arguments.ssd_image)
        except Exception as error:  # every reason it cannot be read is a usage error
            parser.error(str(error))
    return Request(
        label=label,
        search_budgets=[int(value * GIB) for value in requested],
        run_budgets=[int(value * GIB) for value in run],
        physical_capacity=int(max(requested) * GIB),
        spill_budget=int(spill * GIB),
        device=device,
        manual=manual,
        spill=pool,
        image=image,
        external_headroom=int(arguments.external_headroom_gib * GIB),
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
                print(
                    f"  objects under {arguments.minimum_evict_kib} KiB stay"
                    " resident between their uses"
                )
                print(
                    "  graph pairs built per stage:"
                    f" {', '.join(graph_pair_variants(arguments))}"
                )
                if arguments.max_evict_gib is not None:
                    print(
                        f"  write cap: a plan writing more than"
                        f" {arguments.max_evict_gib:g} GiB per step is not run"
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
