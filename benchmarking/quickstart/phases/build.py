"""Build final task definitions, without pools or task measurement."""

from contextlib import nullcontext
from functools import partial
from pathlib import Path

from shadowspill import build_step

from .candidates import attempt
from .common import experiment, write
from .distributed import preparation_size, workers


def run(request):
    setup = experiment(request)
    count = preparation_size(setup)
    # A DP rank map belongs to execution; a local build has no rendezvous.
    binding = request if count > 1 else {**request, "rank_map": None}
    with workers(binding, {"world_size": count}) as group:
        root = Path(request["output_dir"])
        if group.rank == 0:
            (root / "programs" / "index.json").unlink(missing_ok=True)
            (root / "profiled" / "index.json").unlink(missing_ok=True)
        with setup.get("context", nullcontext)():
            _build(request, setup, group)


def _build(request, setup, group):
    root = Path(request["output_dir"])
    model = setup["model_factory"]()
    optimizer = setup["optimizer"](model.parameters())
    entries, world_size = [], 1
    for index, (name, inputs) in enumerate(setup["candidates"].items()):
        if set(inputs) - {"args", "kwargs"}:
            raise ValueError("candidate inputs contain only args/kwargs sequences")
        args, kwargs = inputs.get("args", ()), inputs.get("kwargs", ())
        count = len(args) or len(kwargs)
        if not count or (args and kwargs and len(args) != len(kwargs)):
            raise ValueError(
                "candidate needs equally sized, nonempty args/kwargs sequences"
            )
        print(f"build: {name}, {count} examples", flush=True)
        entry = {
            "name": name,
            "file": f"candidate-{index:04d}.ssp",
            "num_examples": count,
        }
        build = partial(
            build_step,
            model,
            optimizer=optimizer,
            objective=setup["objective"],
            example_calls=list(
                zip(args or [()] * count, kwargs or [{}] * count, strict=True)
            ),
            target=request["device"],
            rank_map=group.rank_map if group.count > 1 else None,
            **setup.get("build_options", {}),
        )
        # A rank-local OOM cannot safely skip a communicating task while peers
        # are inside it. Let torchrun terminate that worker group on failure.
        program = (
            build()
            if group.count > 1
            else attempt(
                entry,
                "build",
                build,
                request["device"],
            )
        )
        if program is not None:
            if group.rank == 0:
                program.save(root / "programs" / entry["file"])
            world_size = (
                1 if program.parallelism is None else program.parallelism.world_size
            )
        entries.append(entry)
        del program
    if not entries:
        raise ValueError("quickstart requires at least one candidate")
    if group.rank != 0:
        return
    write(
        root / "programs" / "index.json",
        {
            "candidates": entries,
            "world_size": world_size,
            "preparation_world_size": group.count,
            "metadata": setup.get("metadata", {}),
            "units_per_step": setup.get("units_per_step"),
            "unit_label": setup.get("unit_label", "units"),
        },
    )
