"""The geometry of one optimizer update, and the tier each round starts from.

An update is ``tokens_per_step`` tokens across every rank, consumed as
``world_size`` ranks times ``accumulation`` microbatches of
``tokens_per_microbatch`` tokens each, in whole sequences of
``sequence_length``. A harness never changes any of these to make a
configuration fit: a geometry that does not divide is refused, and one that
does not fit the device is a row that says so.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass

#: The sequence length every round uses unless a scenario names another.
SEQUENCE_LENGTH = 4_096

#: Tokens per optimizer update each hardware tier starts from: a 32 GB device
#: at the top of the range a single consumer GPU trains at, one H100, and
#: an eight-H100 node.
TIER_TOKENS_PER_STEP = {
    "rtx5090": 131_072,
    "h100": 262_144,
    "h100x8": 2_097_152,
}


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

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")
        accumulation_steps(
            self.tokens_per_step,
            self.sequence_length,
            self.tokens_per_microbatch,
            self.world_size,
        )

    @property
    def accumulation(self) -> int:
        """Microbatches per rank per update."""
        return accumulation_steps(
            self.tokens_per_step,
            self.sequence_length,
            self.tokens_per_microbatch,
            self.world_size,
        )

    @property
    def sequences_per_microbatch(self) -> int:
        return self.tokens_per_microbatch // self.sequence_length

    @property
    def tokens_per_rank_per_step(self) -> int:
        return self.tokens_per_step // self.world_size

    def as_dict(self) -> dict[str, int]:
        return {
            **asdict(self),
            "accumulation": self.accumulation,
            "sequences_per_microbatch": self.sequences_per_microbatch,
            "tokens_per_rank_per_step": self.tokens_per_rank_per_step,
        }

    @classmethod
    def tier(
        cls,
        name: str,
        *,
        tokens_per_microbatch: int,
        world_size: int = 1,
        sequence_length: int = SEQUENCE_LENGTH,
    ) -> Geometry:
        """The tier's tokens per update at the given microbatch."""
        if name not in TIER_TOKENS_PER_STEP:
            raise ValueError(
                f"unknown tier {name!r}; tiers are {', '.join(TIER_TOKENS_PER_STEP)}"
            )
        return cls(
            sequence_length,
            TIER_TOKENS_PER_STEP[name],
            tokens_per_microbatch,
            world_size,
        )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The geometry flags every harness takes, with the tier default."""

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=SEQUENCE_LENGTH,
        help=f"tokens per sequence (default: {SEQUENCE_LENGTH})",
    )
    parser.add_argument(
        "--tokens-per-step",
        type=int,
        default=None,
        help="tokens per optimizer update across every rank; --tier's value"
        " when not given",
    )
    parser.add_argument(
        "--tokens-per-microbatch",
        type=int,
        required=True,
        help="tokens per microbatch per rank, in whole sequences",
    )
    parser.add_argument(
        "--tier",
        choices=tuple(TIER_TOKENS_PER_STEP),
        default=None,
        help="the hardware tier whose tokens per update to use when"
        " --tokens-per-step is not given",
    )


def from_arguments(arguments: argparse.Namespace, world_size: int) -> Geometry:
    """The geometry the flags describe; a step total needs one of its two
    flags, and both together must agree."""

    tokens_per_step = arguments.tokens_per_step
    if arguments.tier is not None:
        tier_tokens = TIER_TOKENS_PER_STEP[arguments.tier]
        if tokens_per_step is None:
            tokens_per_step = tier_tokens
        elif tokens_per_step != tier_tokens:
            raise ValueError(
                f"--tokens-per-step {tokens_per_step} contradicts --tier"
                f" {arguments.tier} ({tier_tokens})"
            )
    if tokens_per_step is None:
        raise ValueError("--tokens-per-step or --tier is required")
    return Geometry(
        arguments.sequence_length,
        tokens_per_step,
        arguments.tokens_per_microbatch,
        world_size,
    )


__all__ = [
    "SEQUENCE_LENGTH",
    "TIER_TOKENS_PER_STEP",
    "Geometry",
    "accumulation_steps",
    "add_arguments",
    "from_arguments",
    "recompute_layers",
]
