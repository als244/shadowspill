"""The catalog: every supplied model by name.

A name is ``<family>-<size>``: ``llama3-8b`` is Llama 3 at its published 8B
shape, ``llama3-numerical`` the small exact-scale shape the numerical gate
runs, ``llama3-tiny`` a shape for smoke tests; a family whose one published
model carries no size in its name keeps that name (``glm53``). The family
package is ``workloads.models.<family>``: its ``config`` module holds
``Config`` and ``PRESETS``, the configurations by size, and
``implementations`` holds one module per implementation, each exposing
``Model``.

An entry also records which implementations build it, the geometry a
benchmark of it starts from, whether its objective carries a router
balancing term, whether the mlops implementation can shard its experts over
an expert-parallel group and, for a checkpoint-bound family, the quickstart
factory that builds it. Adding a model is one family package and one entry
here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from importlib import import_module
from typing import Any

from workloads.providers import ModelImplementation

#: Every implementation a family may supply, in the order the gates name them.
IMPLEMENTATIONS: tuple[ModelImplementation, ...] = ("pytorch", "mlops")


@dataclass(frozen=True, slots=True)
class Geometry:
    """Where a benchmark of a model starts: the sequence length, the tokens
    one microbatch holds and the tokens one optimizer step consumes."""

    sequence_length: int
    tokens_per_microbatch: int
    tokens_per_step: int

    def __post_init__(self) -> None:
        if self.tokens_per_microbatch % self.sequence_length:
            raise ValueError("the sequence length must divide the microbatch")
        if self.tokens_per_step % self.tokens_per_microbatch:
            raise ValueError("the microbatch must divide the step")

    @property
    def sequences_per_microbatch(self) -> int:
        return self.tokens_per_microbatch // self.sequence_length

    @property
    def accumulation_count(self) -> int:
        return self.tokens_per_step // self.tokens_per_microbatch


@dataclass(frozen=True, slots=True)
class Entry:
    """One catalog model."""

    name: str
    family: str
    size: str
    implementations: tuple[ModelImplementation, ...]
    geometry: Geometry
    mixture: bool = False
    expert_parallel: bool = False
    factory: str | None = None
    factory_arguments: Mapping[str, Any] = field(default_factory=dict)
    description: str = ""

    def config(self, **changes: Any) -> Any:
        """The family's ``Config`` at this size, with ``changes`` in place."""

        presets = import_module(f"workloads.models.{self.family}.config").PRESETS
        try:
            preset = presets[self.size]
        except KeyError as error:
            raise ValueError(
                f"{self.family} has no {self.size!r} preset; it has {list(presets)}"
            ) from error
        try:
            return replace(preset, **changes)
        except TypeError as error:
            raise ValueError(f"{self.name}: {error}") from error

    def model_class(self, implementation: str) -> Any:
        """The ``Model`` class of one of this entry's implementations."""

        if implementation not in self.implementations:
            raise ValueError(
                f"{self.name} has no {implementation!r} implementation; "
                f"it has {', '.join(self.implementations)}"
            )
        return import_module(
            f"workloads.models.{self.family}.implementations.{implementation}"
        ).Model


def _text(tokens_per_microbatch: int) -> Geometry:
    """The retained text geometry: 1,024-token sequences, 65,536 tokens a
    step, and the microbatch a family's published size runs at."""

    return Geometry(1_024, tokens_per_microbatch, 65_536)


_BOTH = IMPLEMENTATIONS
_MLOPS: tuple[ModelImplementation, ...] = ("mlops",)
_GLM = "workloads.models.glm53.quickstart:experiment"
_GLM_FLASH = "workloads.models.glm53_flash.quickstart:experiment"

ENTRIES: tuple[Entry, ...] = (
    Entry(
        "llama3-numerical",
        "llama3",
        "numerical",
        _BOTH,
        _text(8_192),
        description="the exact-scale shape the numerical gate runs",
    ),
    Entry(
        "llama3-tiny",
        "llama3",
        "tiny",
        _BOTH,
        _text(4_096),
        description="a smoke-test shape",
    ),
    Entry(
        "llama3-1b",
        "llama3",
        "1b",
        _BOTH,
        _text(8_192),
        description="Llama-3.2-1B's decoder, with tied embeddings",
    ),
    Entry(
        "llama3-8b",
        "llama3",
        "8b",
        _BOTH,
        _text(8_192),
        description="Llama-3.1-8B's decoder",
    ),
    Entry(
        "qwen35-numerical",
        "qwen35",
        "numerical",
        _BOTH,
        _text(16_384),
        description="the exact-scale shape the numerical gate runs",
    ),
    Entry(
        "qwen35-tiny",
        "qwen35",
        "tiny",
        _BOTH,
        _text(4_096),
        description="a smoke-test shape",
    ),
    Entry(
        "qwen35-2b",
        "qwen35",
        "2b",
        _BOTH,
        _text(16_384),
        description="Qwen3.5-2B's text decoder, with tied embeddings",
    ),
    Entry(
        "qwen35-9b",
        "qwen35",
        "9b",
        _BOTH,
        _text(16_384),
        description="Qwen3.5-9B's text decoder",
    ),
    Entry(
        "qwen35-27b",
        "qwen35",
        "27b",
        _BOTH,
        _text(8_192),
        description="Qwen3.5-27B's text decoder, without its"
        " multi-token prediction layer",
    ),
    Entry(
        "olmoe-numerical",
        "olmoe",
        "numerical",
        _BOTH,
        _text(32_768),
        mixture=True,
        expert_parallel=True,
        description="the exact-scale shape the numerical gate runs",
    ),
    Entry(
        "olmoe-tiny",
        "olmoe",
        "tiny",
        _BOTH,
        _text(4_096),
        mixture=True,
        expert_parallel=True,
        description="a smoke-test shape",
    ),
    Entry(
        "olmoe-7b-a1b",
        "olmoe",
        "7b-a1b",
        _BOTH,
        _text(32_768),
        mixture=True,
        expert_parallel=True,
        description="OLMoE-1B-7B",
    ),
    Entry(
        "qwen3moe-tiny",
        "qwen3moe",
        "tiny",
        _MLOPS,
        _text(4_096),
        mixture=True,
        expert_parallel=True,
        description="a smoke-test shape",
    ),
    Entry(
        "qwen3moe-30b-a3b",
        "qwen3moe",
        "30b-a3b",
        _MLOPS,
        _text(8_192),
        mixture=True,
        expert_parallel=True,
        description="Qwen3-30B-A3B",
    ),
    Entry(
        "qwen35moe-tiny",
        "qwen35moe",
        "tiny",
        _MLOPS,
        _text(4_096),
        mixture=True,
        expert_parallel=True,
        description="a smoke-test shape",
    ),
    Entry(
        "qwen35moe-35b-a3b",
        "qwen35moe",
        "35b-a3b",
        _MLOPS,
        _text(8_192),
        mixture=True,
        expert_parallel=True,
        description="Qwen3.5-35B-A3B's text decoder",
    ),
    Entry(
        "glm53",
        "glm53",
        "published",
        _MLOPS,
        _text(8_192),
        mixture=True,
        factory=_GLM,
        description="GLM-5.3, LoRA over a local checkpoint through its factory",
    ),
    Entry(
        "glm53-tiny",
        "glm53",
        "tiny",
        _MLOPS,
        _text(4_096),
        mixture=True,
        factory=_GLM,
        factory_arguments={"tiny": True},
        description="GLM-5.3's small random fixture",
    ),
    Entry(
        "glm53-flash",
        "glm53_flash",
        "published",
        _MLOPS,
        _text(8_192),
        mixture=True,
        factory=_GLM_FLASH,
        description="GLM-5.3-Flash, LoRA over a local checkpoint through its factory",
    ),
    Entry(
        "glm53-flash-tiny",
        "glm53_flash",
        "tiny",
        _MLOPS,
        _text(4_096),
        mixture=True,
        factory=_GLM_FLASH,
        factory_arguments={"tiny": True},
        description="GLM-5.3-Flash's small random fixture",
    ),
)

_BY_NAME: Mapping[str, Entry] = {item.name: item for item in ENTRIES}
assert len(_BY_NAME) == len(ENTRIES), "catalog names repeat"


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


def build_on_meta(
    model: str,
    implementation: ModelImplementation = "mlops",
    *,
    dtype: str | None = None,
    lora: Mapping[str, Any] | None = None,
    **changes: Any,
) -> Any:
    """A catalog model declared on ``meta``: structure in ``dtype``, no
    values, LoRA applied when ``lora`` names its options, ``changes`` on
    the entry's configuration. The form a saved training recipe names."""

    from .construction import build_on_meta as declare

    item = entry(model)
    return declare(
        item.model_class(implementation),
        dtype=dtype,
        lora=None if lora is None else dict(lora),
        config=item.config(**changes),
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
