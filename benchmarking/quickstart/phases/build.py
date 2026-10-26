"""Build final task definitions, without pools or task measurement."""

from pathlib import Path

from shadowspill import build_step

from .common import experiment, write


def run(request):
    setup = experiment(request)
    model = setup["model_factory"]()
    optimizer = setup["optimizer"](model.parameters())
    root = Path(request["output_dir"]) / "programs"
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
        program = build_step(
            model,
            optimizer=optimizer,
            objective=setup["objective"],
            example_calls=list(
                zip(args or [()] * count, kwargs or [{}] * count, strict=True)
            ),
            target=request["device"],
            **setup.get("build_options", {}),
        )
        filename = f"candidate-{index:04d}.ssp"
        program.save(root / filename)
        world_size = (
            1 if program.parallelism is None else program.parallelism.world_size
        )
        entries.append({"name": name, "file": filename, "num_examples": count})
    if not entries:
        raise ValueError("quickstart requires at least one candidate")
    write(
        root / "index.json",
        {
            "candidates": entries,
            "world_size": world_size,
            "metadata": setup.get("metadata", {}),
            "units_per_step": setup.get("units_per_step"),
            "unit_label": setup.get("unit_label", "units"),
        },
    )
