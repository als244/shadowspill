"""CPU loader descriptions; bounded random/pretrained work happens in workers."""

from contextlib import contextmanager
from dataclasses import asdict

from shadowspill.training.definitions import ModelInitialization

from . import hub_files
from .hub import immutable_revision


def normalize_files(values):
    return (
        None if values is None else [asdict(file) for file in hub_files.files(values)]
    )


def describe(family, config, context, weight_files):
    weights = context.weights
    width = getattr(config, "rotary_width", config.head_dim)
    if weights.kind == "random":
        if weight_files is not None:
            raise ValueError(
                "random initialization cannot name pretrained weight files"
            )

        @contextmanager
        def open_random(runtime):
            from .text import Initialize

            yield Initialize(family, config, seed=weights.seed)

        return ModelInitialization(
            peak_bytes=(32 << 20) + 32 * width,
            chunk_bytes=1 << 20,
            open=open_random,
        )
    if weights.kind != "pretrained" or not immutable_revision(weights.revision):
        raise ValueError("catalog initialization requires random or pinned Hub weights")
    manifest = hub_files.files(weight_files)

    @contextmanager
    def open_pretrained(runtime):
        from .headers import validate
        from .text import FromCheckpoint

        with hub_files.snapshot(
            weights.reference, weights.revision, manifest, runtime
        ) as directory:
            validate(directory, manifest)
            with FromCheckpoint(family, directory, **asdict(config)) as loader:
                yield loader

    # Metadata limits: 1 MiB config, 4 MiB index, 4 MiB aggregate headers,
    # <=65536 fixed-schema tensor declarations, rank<=8 and bounded key strings.
    # Includes decoding/retained reader state, 1 MiB HTTP/hash/destination chunks,
    # mapped dtype conversion pages and the existing deterministic generator.
    return ModelInitialization(
        peak_bytes=(224 << 20) + 32 * width,
        chunk_bytes=1 << 20,
        disk_bytes=hub_files.disk_bytes(manifest),
        open=open_pretrained,
    )
