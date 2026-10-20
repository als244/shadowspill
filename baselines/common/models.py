"""The catalog model a harness builds, in full or LoRA mode, with the
objective every harness shares and the useful work a step performs.

A ``ModelRequest`` names a catalog entry, the implementation that builds
it, the training mode and, for LoRA, the adapter options ``configure_lora``
takes; it builds the model on ``meta`` so each harness materializes it where
its sharding wants, and it supplies the objective of ``workloads.full_model``
(each microbatch's summed loss over the rank's tokens per update), so every
harness optimizes the same quantity.

Useful work is counted from the model that was built, not from a table:
every two-dimensional weight is a matrix multiply of its size per token, a
routed expert bank counts ``top_k`` of its experts, embeddings count
nothing, and full-attention layers add their score and value products at
the sequence length. Forward is two FLOPs per weight element, the backward
two for the input gradient and, for a trained weight, two more for its
gradient; recomputation, the optimizer and communication are not credited,
so a configuration that recomputes reports the same useful work at a longer
step. Causal masking is not credited either.
"""

from __future__ import annotations

import argparse
import inspect
import json
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from workloads.models import catalog

from .geometry import Geometry

MODES = ("full", "lora")


@dataclass(frozen=True)
class ModelRequest:
    model: str
    implementation: str = "mlops"
    mode: str = "full"
    lora: Mapping[str, Any] = field(default_factory=dict)
    config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        entry = catalog.entry(self.model)
        if self.implementation not in entry.implementations:
            raise ValueError(
                f"{self.model} has no {self.implementation!r} implementation;"
                f" it has {', '.join(entry.implementations)}"
            )
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if entry.factory is not None:
            raise ValueError(
                f"{self.model} is built by its quickstart factory, which the"
                " baseline harnesses do not run"
            )
        entry.config(**dict(self.config))
        if self.lora:
            from workloads.lora import configure_lora

            allowed = set(inspect.signature(configure_lora).parameters) - {"model"}
            unknown = set(self.lora) - allowed
            if unknown:
                raise ValueError(f"unknown LoRA options {sorted(unknown)}")
        if self.lora and self.mode != "lora":
            raise ValueError("LoRA options need mode=lora")

    @property
    def entry(self) -> catalog.Entry:
        return catalog.entry(self.model)

    def checkpoint(self, directory: str, **overrides: Any) -> Any:
        """The family's reader open on a local Hugging Face checkpoint
        (``workloads.models.<family>.checkpoint``), whose ``config`` must be
        this request's configuration apart from the rotary capacity."""

        import importlib
        from dataclasses import replace

        module = importlib.import_module(
            f"workloads.models.{self.entry.family}.checkpoint"
        )
        readers = [
            value
            for name, value in vars(module).items()
            if name.endswith("Checkpoint")
            and isinstance(value, type)
            and value.__module__ == module.__name__
        ]
        if len(readers) != 1:
            raise ValueError(f"{self.entry.family} has no checkpoint reader")
        config = self.entry.config(**dict(self.config))
        reader = readers[0](directory, max_seq_len=config.max_seq_len, **overrides)
        if replace(reader.config, max_seq_len=config.max_seq_len) != config:
            reader.close()
            raise ValueError(
                f"{directory} holds a different {self.entry.family} than"
                f" {self.model}: {reader.config} is not {config}"
            )
        return reader

    def spec(self, geometry: Geometry) -> Any:
        """The ``FullModelSpec`` of this model at the rank's geometry: the
        objective normalizes by the rank's tokens per update."""

        from workloads.full_model import spec

        return replace(
            spec(self.model, self.implementation, **dict(self.config)),
            sequence_length=geometry.sequence_length,
            sequences_per_microbatch=geometry.sequences_per_microbatch,
            accumulation_count=geometry.accumulation,
        )

    def build(self, *, dtype: str, device: str = "meta") -> Any:
        """The model on ``meta`` (or built with values on ``device``) at
        ``dtype``, with LoRA applied in that mode so the trainable set is
        fixed before any harness shards or materializes it; the factors are
        created at ``dtype`` too, unless the LoRA options say otherwise, so
        they follow the same precision contract as every other trained
        weight."""

        lora = None
        if self.mode == "lora":
            lora = {"factor_dtype": dtype, **self.lora}
        return catalog.build_on_meta(
            self.model,
            self.implementation,
            dtype=dtype,
            lora=lora,
            device=device,
            **dict(self.config),
        )

    def objective(self, geometry: Geometry) -> Any:
        """``objective(model, tokens, targets, sequence_lengths)``: the
        microbatch's summed loss divided by the rank's tokens per update."""

        import functools

        from workloads.full_model import full_model_objective

        return functools.partial(full_model_objective, self.spec(geometry))

    def summary(self, model: Any) -> dict[str, Any]:
        """What was built: counts of trained and frozen parameters, tying,
        and the mode."""

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        head = getattr(model, "lm_head", None)
        embed = getattr(model, "embed", None)
        tied = (
            head is not None
            and embed is not None
            and getattr(head, "weight", None) is getattr(embed, "weight", None)
        )
        return {
            "model": self.model,
            "implementation": self.implementation,
            "family": self.entry.family,
            "mode": self.mode,
            "lora": dict(self.lora),
            "config": dict(self.config),
            "mixture": self.entry.mixture,
            "parameters": total,
            "trainable_parameters": trainable,
            "tied_embeddings": bool(tied),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "implementation": self.implementation,
            "mode": self.mode,
            "lora": dict(self.lora),
            "config": dict(self.config),
        }


def linear_work(model: Any) -> dict[str, float]:
    """Weight elements multiplied per token, frozen and trained: every
    weight of two dimensions or more counts its elements, a routed bank
    (first dimension the expert count) counts ``top_k`` experts of itself,
    and embedding tables count nothing."""

    import torch

    config = model.config
    experts = getattr(config, "n_experts", None)
    top_k = getattr(config, "top_k", None)
    embedded = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, torch.nn.Embedding)
        for parameter in module.parameters(recurse=False)
    }
    head_weight = getattr(getattr(model, "lm_head", None), "weight", None)
    work = {"frozen": 0.0, "trainable": 0.0}
    for parameter in model.parameters():
        if parameter.ndim < 2:
            continue
        if id(parameter) in embedded and parameter is not head_weight:
            continue
        elements = float(parameter.numel())
        if experts and top_k and parameter.ndim >= 3 and parameter.shape[0] == experts:
            elements *= top_k / experts
        work["trainable" if parameter.requires_grad else "frozen"] += elements
    return work


def full_attention_layers(model: Any) -> int:
    """Blocks whose mixer attends over the sequence; a block that names its
    kind is counted by it, any other block is full attention."""

    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise ValueError("the model has no blocks to count attention over")
    return sum(1 for block in blocks if getattr(block, "kind", "full") == "full")


def useful_flops_per_token(model: Any, sequence_length: int) -> float:
    """The convention in the module docstring, for the model as built."""

    if sequence_length < 1:
        raise ValueError("sequence length must be positive")
    config = model.config
    width = int(config.n_heads) * int(config.head_dim)
    work = linear_work(model)
    linear = 4.0 * (work["frozen"] + work["trainable"]) + 2.0 * work["trainable"]
    attention = 12.0 * sequence_length * width * full_attention_layers(model)
    return linear + attention


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The model flags every harness takes."""

    parser.add_argument("--model", required=True, choices=catalog.names())
    parser.add_argument(
        "--implementation", choices=catalog.IMPLEMENTATIONS, default="mlops"
    )
    parser.add_argument("--mode", choices=MODES, default="full")
    parser.add_argument(
        "--lora",
        type=json.loads,
        default={},
        metavar="JSON",
        help="configure_lora options for mode=lora, such as"
        ' \'{"rank": 16, "alpha": 16, "head": "frozen"}\'',
    )
    parser.add_argument(
        "--model-config",
        type=json.loads,
        default={},
        metavar="JSON",
        help="field overrides on the catalog entry's configuration",
    )


def from_arguments(arguments: argparse.Namespace) -> ModelRequest:
    return ModelRequest(
        arguments.model,
        arguments.implementation,
        arguments.mode,
        dict(arguments.lora),
        dict(arguments.model_config),
    )


__all__ = [
    "MODES",
    "ModelRequest",
    "add_arguments",
    "from_arguments",
    "full_attention_layers",
    "linear_work",
    "useful_flops_per_token",
]
