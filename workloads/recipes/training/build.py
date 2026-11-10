"""Build adapter-supplied candidates with the ordinary public frontend."""

import json
import sys

from benchmarking.quickstart.phases.distributed import workers
from shadowspill import build_forward, build_step
from workloads.recipes.phases import settings

from .candidates import attempt_candidate
from .configuration import adapter


def build_candidate(root, options, candidate, definition, group):
    configured = {**options, **candidate}
    source = definition.inputs(configured, rank=group.logical_rank)
    examples = source.example()
    count = len(examples["args"])
    if count != candidate["num_examples"]:
        raise ValueError("candidate example count differs from its input source")
    with definition.setup(configured, rank=group.logical_rank) as setup:
        print(f"building {candidate['path']}", flush=True)
        build_options = dict(setup.build_options)
        if group.count > 1:
            build_options["rank_map"] = group.rank_map
        training = build_step(
            setup.model,
            optimizer=setup.optimizer,
            objective=setup.objective,
            example_args=examples["args"][0],
            example_kwargs=examples["kwargs"][0],
            num_examples=count,
            **build_options,
        )
        at = root / "programs" / candidate["path"]
        if group.rank == 0:
            training.save(at / "training.ssp")
            (at / "metadata.json").write_text(
                json.dumps(setup.metadata, indent=2) + "\n"
            )
        if options["eval_every"]:
            evaluation = build_forward(
                setup.model.eval(),
                forward_fn=setup.objective,
                example_args=examples["args"][0],
                example_kwargs=examples["kwargs"][0],
                **{
                    key: value
                    for key, value in build_options.items()
                    if key
                    not in {
                        "graph_pair_policy",
                        "master_dtype",
                        "grad_dtype",
                        "fuse_narrow_gradient_accumulation",
                        "gradient_destination_rules",
                    }
                },
            )
            if group.rank == 0:
                evaluation.save(at / "evaluation.ssp")
    return True


def run(directory):
    root, options = settings(directory)
    definition = adapter(options["recipe"])
    with workers({}, {"world_size": options["preparation_ranks"]}) as group:
        for candidate in options["candidates"]:
            attempt_candidate(
                root,
                candidate,
                "build",
                lambda candidate=candidate: build_candidate(
                    root, options, candidate, definition, group
                ),
                group,
            )


if __name__ == "__main__":
    run(sys.argv[1])
