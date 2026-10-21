"""The geometry of one optimizer update.

An update is ``tokens_per_step`` tokens across every rank, consumed as
``replicas`` data-parallel replicas times ``accumulation`` microbatches of
``tokens_per_microbatch`` tokens each, in whole sequences of
``sequence_length``. Every rank is a replica unless a system splits one
replica over several ranks (tensor, pipeline or context parallelism), which
names the replicas its ``world_size`` ranks form. An experiment chooses
them for its model, sequence length and hardware together. A harness never
changes any of these to make a configuration fit: a geometry that does not
divide is refused, and one that does not fit the device is a row that says
so.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass

#: The sequence length of a case that names none.
SEQUENCE_LENGTH = 4_096


def accumulation_steps(global_tokens, sequence_length, microbatch_tokens, world_size):
    if min(global_tokens, sequence_length, microbatch_tokens, world_size) < 1:
        raise ValueError("token counts and world size must be positive")
    if microbatch_tokens % sequence_length:
        raise ValueError("microbatch tokens must contain whole sequences")
    local_step, remainder = divmod(global_tokens, world_size)
    if remainder or local_step % microbatch_tokens:
        raise ValueError("global tokens must divide into complete rank microbatches")
    return local_step // microbatch_tokens


def recompute_layers(layer_count, fraction):
    """Spread the selected checkpointed layers evenly through the model."""
    if layer_count < 1 or not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError("recompute fraction must be in [0, 1]")
    count = round(layer_count * fraction)
    return tuple(
        i
        for i in range(layer_count)
        if (i + 1) * count // layer_count > i * count // layer_count
    )


@dataclass(frozen=True)
class Geometry:
    """One optimizer update's shape; every field is an integer count."""

    sequence_length: int
    tokens_per_step: int
    tokens_per_microbatch: int
    world_size: int = 1
    #: Data-parallel replicas, when fewer than the ranks.
    replicas: int | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int and not (name == "replicas" and value is None):
                raise ValueError(f"{name} must be an integer")
        if self.data_parallel < 1 or self.world_size % self.data_parallel:
            raise ValueError("the replicas must divide the world")
        accumulation_steps(
            self.tokens_per_step,
            self.sequence_length,
            self.tokens_per_microbatch,
            self.data_parallel,
        )

    @property
    def data_parallel(self) -> int:
        """Data-parallel replicas: every rank, unless ``replicas`` says."""
        return self.world_size if self.replicas is None else self.replicas

    @property
    def accumulation(self) -> int:
        """Microbatches per replica per update."""
        return accumulation_steps(
            self.tokens_per_step,
            self.sequence_length,
            self.tokens_per_microbatch,
            self.data_parallel,
        )

    @property
    def sequences_per_microbatch(self) -> int:
        return self.tokens_per_microbatch // self.sequence_length

    @property
    def tokens_per_rank_per_step(self) -> int:
        """Tokens per replica per update, which every rank of it reads."""
        return self.tokens_per_step // self.data_parallel

    def as_dict(self) -> dict[str, int]:
        value = asdict(self)
        if self.replicas is None:
            del value["replicas"]
        return {
            **value,
            "accumulation": self.accumulation,
            "sequences_per_microbatch": self.sequences_per_microbatch,
            "tokens_per_rank_per_step": self.tokens_per_rank_per_step,
        }


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The geometry flags every harness takes."""

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=SEQUENCE_LENGTH,
        help=f"tokens per sequence (default: {SEQUENCE_LENGTH})",
    )
    parser.add_argument(
        "--tokens-per-step",
        type=int,
        required=True,
        help="tokens per optimizer update across every rank",
    )
    parser.add_argument(
        "--tokens-per-microbatch",
        type=int,
        required=True,
        help="tokens per microbatch per rank, in whole sequences",
    )


def from_arguments(arguments: argparse.Namespace, world_size: int) -> Geometry:
    """The geometry the flags describe."""

    return Geometry(
        arguments.sequence_length,
        arguments.tokens_per_step,
        arguments.tokens_per_microbatch,
        world_size,
    )


__all__ = [
    "SEQUENCE_LENGTH",
    "Geometry",
    "accumulation_steps",
    "add_arguments",
    "from_arguments",
    "recompute_layers",
]
