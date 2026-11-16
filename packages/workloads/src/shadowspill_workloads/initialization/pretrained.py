"""CPU loader descriptions; bounded random/pretrained work happens in workers."""

from contextlib import contextmanager
from dataclasses import asdict

from shadowspill.training.definitions import ModelInitialization

from . import hub_files, local_files, weight_files
from .hub import immutable_revision


def describe(family, config, context):
    weights = context.weights
    lora_seed = context.model.config.get("lora_seed")
    width = getattr(config, "rotary_width", config.head_dim)
    if context.temporary is not None:
        if not set(context.temporary).issubset({"rotary.cosine", "rotary.sine"}):
            raise ValueError(
                "catalog temporary initialization requires derived rotary buffers"
            )
        seed = 0  # These tables depend only on model geometry, never on weight RNG.
    else:
        seed = weights.seed
    if context.temporary is not None or weights.kind == "random":

        @contextmanager
        def open_random(runtime):
            from .decoder import Initialize

            yield Initialize(family, config, seed=seed, lora_seed=lora_seed)

        return ModelInitialization(
            peak_bytes=(32 << 20) + 32 * width,
            chunk_bytes=1 << 20,
            open=open_random,
        )
    if weights.kind == "pretrained" and local_files.accepts(weights.reference):
        if set(weights.metadata) != {"files"}:
            raise ValueError("local checkpoint requires a resolved file manifest")
        manifest = weight_files.files(weights.metadata["files"])
        if weights.revision != local_files.revision(manifest):
            raise ValueError("local checkpoint manifest differs from its revision")

        @contextmanager
        def open_local(runtime):
            from .decoder import FromCheckpoint

            with (
                local_files.snapshot(weights.reference, manifest, runtime) as directory,
                FromCheckpoint(
                    family, directory, lora_seed=lora_seed, **asdict(config)
                ) as loader,
            ):
                yield loader

        return ModelInitialization(
            peak_bytes=(224 << 20) + 32 * width,
            chunk_bytes=1 << 20,
            open=open_local,
            disk_bytes=local_files.disk_bytes(manifest),
        )
    if weights.kind != "pretrained" or not immutable_revision(weights.revision):
        raise ValueError("catalog initialization requires random or pinned Hub weights")
    if set(weights.metadata) != {"files"}:
        raise ValueError("pinned Hub weights require a resolved file manifest")
    manifest = weight_files.files(weights.metadata["files"])

    @contextmanager
    def open_pretrained(runtime):
        from .decoder import FromCheckpoint
        from .headers import validate

        with hub_files.snapshot(
            weights.reference, weights.revision, manifest, runtime
        ) as directory:
            validate(directory, manifest)
            with FromCheckpoint(
                family, directory, lora_seed=lora_seed, **asdict(config)
            ) as loader:
                yield loader

    # Metadata limits: 1 MiB config, 4 MiB index, 4 MiB aggregate headers,
    # <=65536 fixed-schema tensor declarations, rank<=8 and bounded key strings.
    # Includes decoding/retained reader state, 1 MiB HTTP/hash/destination chunks,
    # mapped dtype conversion pages and the existing deterministic generator.
    return ModelInitialization(
        peak_bytes=(224 << 20) + 32 * width,
        chunk_bytes=1 << 20,
        disk_bytes=weight_files.disk_bytes(manifest),
        open=open_pretrained,
    )
