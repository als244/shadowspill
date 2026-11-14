"""Build adapter-supplied candidates with the ordinary public frontend."""

import sys

from shadowspill import build_forward, build_step
from shadowspill.orchestration.launch.workers import workers
from shadowspill.training.preparation.phases import settings

from .build_reuse import check_metadata, completed, save_metadata
from .candidates import attempt_candidate, available
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
        at = root / "programs" / candidate["path"]
        check_metadata(at / "metadata.json", setup.metadata, group)
        build_options = dict(setup.build_options)
        if group.count > 1:
            build_options["rank_map"] = group.rank_map
        if not completed(at / "training.ssp", group, kind="step", num_examples=count):
            training = build_step(
                setup.model,
                optimizer=setup.optimizer,
                objective=setup.objective,
                example_args=examples["args"][0],
                example_kwargs=examples["kwargs"][0],
                num_examples=count,
                **build_options,
            )
            if group.rank == 0:
                training.save(at / "training.ssp")
        if group.rank == 0:
            save_metadata(at / "metadata.json", setup.metadata)
        if options["eval_every"] and not completed(
            at / "evaluation.ssp", group, kind="forward"
        ):
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
    with workers(options["preparation_ranks"]) as group:
        for candidate in available(root, options):
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
