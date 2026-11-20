"""Describe bounded random/checkpoint loading through the common worker API."""

from contextlib import contextmanager
from dataclasses import replace
from importlib import import_module

from shadowspill.training.components import ModelInitialization
from shadowspill_workloads.initialization import hub_files, local_files, weight_files


def describe(config, context):
    if config.implementation == "mlops_ep":
        from .expert_parallel import ownership

        ownership(config, context.parallelism)
        base = describe(
            replace(config, implementation="mlops", expert_options=None), context
        )

        @contextmanager
        def open_experts(runtime):
            from shadowspill_workloads.models.shared.glm.initialization.experts import (
                Initialize,
            )

            ranks = context.parallelism.ranks_per_replica
            with base.open(runtime) as loader:
                yield Initialize(loader, rank=context.logical_rank % ranks, ranks=ranks)

        return replace(base, open=open_experts)
    weights = context.weights
    seed = config.lora_seed if config.lora_seed is not None else weights.seed or 0

    @contextmanager
    def open_random(runtime):
        import torch

        from shadowspill_workloads.models.shared.glm.initialization.loader import (
            Initialize,
        )

        from .configuration import create

        with torch.device("meta"):
            model = create(config)
        yield Initialize(model, seed=seed)

    if context.temporary is not None or weights.kind == "random":
        return ModelInitialization(
            peak_bytes=64 << 20, chunk_bytes=1 << 20, open=open_random
        )
    if weights.kind != "pretrained" or set(weights.metadata) != {"files"}:
        raise ValueError(
            "GLM initialization requires random or pinned pretrained weights"
        )
    manifest = weight_files.files(weights.metadata["files"])
    local = local_files.accepts(weights.reference)
    if local and local_files.revision(manifest) != weights.revision:
        raise ValueError("local GLM checkpoint differs from its pinned manifest")

    @contextmanager
    def open_pretrained(runtime):
        import torch

        from shadowspill_workloads.models.shared.glm.initialization.loader import (
            Initialize,
        )

        from .configuration import create

        snapshot = (
            local_files.snapshot(weights.reference, manifest, runtime)
            if local
            else hub_files.snapshot(
                weights.reference, weights.revision, manifest, runtime
            )
        )
        reader = import_module(
            f"shadowspill_workloads.models.{config.family}.checkpoint"
        ).GLMCheckpoint
        with (
            snapshot as directory,
            reader(
                directory, expert_storage=config.architecture["expert_storage"]
            ) as source,
        ):
            source.build_model()
            with torch.device("meta"):
                model = create(config)
            yield Initialize(
                model, source=source, initialization="checkpoint", seed=seed
            )

    return ModelInitialization(
        peak_bytes=1 << 30,
        chunk_bytes=16 << 20,
        open=open_pretrained,
        disk_bytes=local_files.disk_bytes(manifest)
        if local
        else weight_files.disk_bytes(manifest),
    )
