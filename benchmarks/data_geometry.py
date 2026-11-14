"""Canonical data geometry shared by benchmark datasets and evaluations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DataGeometry:
    """Packed-token geometry for one optimizer step."""

    sequence_length: int
    tokens_per_microbatch: int
    accumulation_rounds: int

    def __post_init__(self) -> None:
        for name, value in (
            ("sequence_length", self.sequence_length),
            ("tokens_per_microbatch", self.tokens_per_microbatch),
            ("accumulation_rounds", self.accumulation_rounds),
        ):
            if value <= 0:
                raise ValueError(f"DataGeometry.{name} must be positive")
        if self.tokens_per_microbatch % self.sequence_length:
            raise ValueError(
                "DataGeometry.sequence_length must divide tokens_per_microbatch"
            )

    @property
    def sequences_per_microbatch(self) -> int:
        return self.tokens_per_microbatch // self.sequence_length

    @property
    def tokens_per_optimizer_step(self) -> int:
        return self.tokens_per_microbatch * self.accumulation_rounds

    @property
    def identity_fragment(self) -> str:
        """Stable human-readable identity shared by benchmark artifacts."""

        return (
            f"tokens-{self.tokens_per_microbatch}__"
            f"sequence-{self.sequence_length}__"
            f"accumulation-{self.accumulation_rounds}"
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "sequence_length": self.sequence_length,
            "tokens_per_microbatch": self.tokens_per_microbatch,
            "sequences_per_microbatch": self.sequences_per_microbatch,
            "accumulation_rounds": self.accumulation_rounds,
            "tokens_per_optimizer_step": self.tokens_per_optimizer_step,
        }

    def describe(self) -> str:
        """Return one concise log description of the complete geometry."""

        return (
            f"sequence_length={self.sequence_length}, "
            f"tokens_per_microbatch={self.tokens_per_microbatch}, "
            f"sequences_per_microbatch={self.sequences_per_microbatch}, "
            f"accumulation_rounds={self.accumulation_rounds}, "
            f"tokens_per_optimizer_step={self.tokens_per_optimizer_step}"
        )


__all__ = ["DataGeometry"]
