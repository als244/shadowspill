"""The catalog: every supplied model by name.

A name is ``<family>-<size>``: ``llama3-8b`` is Llama 3 at its published 8B
shape, ``llama3-numerical`` the small exact-scale shape the numerical gate
runs, ``llama3-tiny`` a shape for smoke tests; a family whose one published
model carries no size in its name keeps that name (``glm53``). The family
package is ``workloads.models.<family>``: its ``config`` module holds
``Config`` and ``PRESETS``, the configurations by size, ``implementations``
holds one module per implementation, each exposing ``Model``, and
``checkpoint`` holds the reader of a local Hugging Face checkpoint of the
family, named in ``READERS``.

An entry also records which implementations build it, the geometry a
benchmark of it starts from, whether its objective carries a router
balancing term, whether the mlops implementation can shard its experts over
an expert-parallel group and, for a checkpoint-bound family, the quickstart
factory that builds it. Adding a model is one family package and one entry
here."""

from collections.abc import Mapping
from typing import Any

from workloads.providers import ModelImplementation

from .entry import IMPLEMENTATIONS, Entry, Geometry
from .presets import ENTRIES

_BY_NAME: Mapping[str, Entry] = {item.name: item for item in ENTRIES}
assert len(_BY_NAME) == len(ENTRIES), "catalog names repeat"

#: Each family's checkpoint reader class in ``workloads.models.<family>.checkpoint``;
#: the GLM families load through their own recipe.
READERS: Mapping[str, str] = {
    "llama3": "Llama3Checkpoint",
    "qwen3": "Qwen3Checkpoint",
    "qwen35": "Qwen35Checkpoint",
    "olmoe": "OLMoECheckpoint",
    "qwen3moe": "Qwen3MoECheckpoint",
    "qwen35moe": "Qwen35MoECheckpoint",
}


def entries() -> tuple[Entry, ...]:
    return ENTRIES


def names() -> tuple[str, ...]:
    return tuple(_BY_NAME)


def families() -> tuple[str, ...]:
    return tuple(dict.fromkeys(item.family for item in ENTRIES))


def entry(name: str) -> Entry:
    """The entry called ``name``; a miss names every catalog model."""

    try:
        return _BY_NAME[name]
    except KeyError as error:
        raise ValueError(
            f"unknown model {name!r}; the catalog has {', '.join(_BY_NAME)}"
        ) from error


def expert_parallel_names() -> tuple[str, ...]:
    return tuple(item.name for item in ENTRIES if item.expert_parallel)


def reader(family: str) -> Any:
    """The family's checkpoint reader: ``reader(family)(directory, **changes)``
    opens a local Hugging Face checkpoint of it, ``changes`` over the
    configuration the checkpoint describes."""

    from importlib import import_module

    try:
        name = READERS[family]
    except KeyError as error:
        raise ValueError(
            f"{family!r} has no catalog checkpoint reader; "
            f"readers exist for {', '.join(READERS)}"
        ) from error
    return getattr(import_module(f"workloads.models.{family}.checkpoint"), name)


def build_on_meta(
    model: str,
    implementation: ModelImplementation = "mlops",
    *,
    dtype: str | None = None,
    lora: Mapping[str, Any] | None = None,
    device: str = "meta",
    **changes: Any,
) -> Any:
    """A catalog model declared on ``meta`` (or built with values on another
    ``device``): structure in ``dtype``, LoRA applied when ``lora`` names
    its options, ``changes`` on the entry's configuration. The form a saved
    training recipe names."""

    from ..construction import build_on_meta as declare

    item = entry(model)
    return declare(
        item.model_class(implementation),
        dtype=dtype,
        lora=None if lora is None else dict(lora),
        device=device,
        config=item.config(**changes),
    )


def build_from_checkpoint(
    family: str,
    directory: str,
    implementation: ModelImplementation = "mlops",
    *,
    dtype: str | None = None,
    lora: Mapping[str, Any] | None = None,
    device: str = "meta",
    **changes: Any,
) -> Any:
    """The ``family`` model a local Hugging Face checkpoint describes, declared
    on ``meta`` at the checkpoint's configuration with ``changes`` (such as
    ``max_seq_len``) over it: structure in ``dtype``, LoRA applied when
    ``lora`` names its options. The form a fine-tuning config names, beside
    ``workloads.recipes.text.initialization.from_checkpoint``."""

    from ..construction import build_on_meta as declare

    with reader(family)(directory, **changes) as checkpoint:
        configuration = checkpoint.config
    item = next(item for item in ENTRIES if item.family == family)
    return declare(
        item.model_class(implementation),
        dtype=dtype,
        lora=None if lora is None else dict(lora),
        device=device,
        config=configuration,
    )


__all__ = [
    "ENTRIES",
    "IMPLEMENTATIONS",
    "Entry",
    "Geometry",
    "build_on_meta",
    "entries",
    "entry",
    "expert_parallel_names",
    "families",
    "names",
]
