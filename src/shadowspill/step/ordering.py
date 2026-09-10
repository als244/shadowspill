"""How a training step walks its microbatches."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class StepDataOrdering:
    """Which microbatch runs which stage when, within one accumulated step.

    A step is ``depth`` passes over ``breadth`` microbatches each. Within a
    pass every microbatch runs one stage's forward before any runs the next
    stage's, and the backward walks the stages in reverse the same way, so
    each stage's parameters are fetched once per pass rather than once per
    microbatch. Two flags shape the corners. ``pair_loss`` runs each
    microbatch's last forward stage and that stage's backward together, so
    the loss's saved state is consumed the moment it exists instead of being
    held for the whole pass. ``reverse_breadth`` walks the pass's microbatches
    in reverse during backward, so the freshest activations go first. Both
    are vacuous at ``breadth == 1``, the microbatch-major order.

    The gradient of a stage is created by the first backward the walk emits
    for it and accumulated into by every later one, so which microbatch
    creates it follows from the walk: :meth:`creates` says which.
    """

    depth: int
    breadth: int
    reverse_breadth: bool = True
    pair_loss: bool = True

    def __post_init__(self) -> None:
        for name in ("depth", "breadth"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer, not {value!r}")
        for name in ("reverse_breadth", "pair_loss"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a bool")

    @property
    def microbatches(self) -> int:
        return self.depth * self.breadth

    @property
    def label(self) -> str:
        """``<depth>x<breadth>`` with ``r`` and ``p`` for the flags that are on."""
        return (
            f"{self.depth}x{self.breadth}"
            f"{'r' if self.reverse_breadth else ''}{'p' if self.pair_loss else ''}"
        )

    @classmethod
    def from_label(cls, label: str) -> StepDataOrdering:
        """The ordering a ``<depth>x<breadth>[r][p]`` label names."""
        match = re.fullmatch(r"(\d+)x(\d+)(r?)(p?)", label.strip())
        if match is None:
            raise ValueError(
                f"ordering label {label!r} is not <depth>x<breadth> with optional r, p"
            )
        return cls(
            int(match.group(1)),
            int(match.group(2)),
            reverse_breadth=match.group(3) == "r",
            pair_loss=match.group(4) == "p",
        )

    @classmethod
    def depth_first(
        cls,
        microbatches: int,
        *,
        reverse_breadth: bool = True,
        pair_loss: bool = True,
    ) -> StepDataOrdering:
        """One microbatch per pass: the microbatch-major order."""
        return cls(microbatches, 1, reverse_breadth, pair_loss)

    @classmethod
    def resolve(
        cls,
        *,
        microbatches: int,
        depth: int | None = None,
        breadth: int | None = None,
        reverse_breadth: bool = True,
        pair_loss: bool = True,
    ) -> StepDataOrdering:
        """The ordering a caller asked for, with the count they left out filled in.

        Neither count given means depth-first. One given fixes the other.
        Both given must multiply to the microbatch count; a mismatch is an
        error that names all three numbers, because a step that silently
        dropped or repeated microbatches would train on the wrong data.
        """
        if microbatches < 1:
            raise ValueError("a step needs at least one microbatch")
        if depth is None and breadth is None:
            return cls.depth_first(
                microbatches, reverse_breadth=reverse_breadth, pair_loss=pair_loss
            )
        if depth is None:
            assert breadth is not None
            depth, remainder = divmod(microbatches, breadth) if breadth > 0 else (0, 1)
            if remainder or depth < 1:
                raise ValueError(
                    f"breadth {breadth} does not divide the {microbatches} microbatches"
                )
        elif breadth is None:
            breadth, remainder = divmod(microbatches, depth) if depth > 0 else (0, 1)
            if remainder or breadth < 1:
                raise ValueError(
                    f"depth {depth} does not divide the {microbatches} microbatches"
                )
        elif depth * breadth != microbatches:
            raise ValueError(
                f"depth {depth} times breadth {breadth} is {depth * breadth}, "
                f"but the step has {microbatches} microbatches"
            )
        return cls(depth, breadth, reverse_breadth, pair_loss)

    def positions(self, pass_index: int) -> range:
        """The microbatches of one pass, in forward order."""
        if not 0 <= pass_index < self.depth:
            raise IndexError(f"pass {pass_index} of {self.depth}")
        return range(pass_index * self.breadth, (pass_index + 1) * self.breadth)

    def backward_positions(self, pass_index: int) -> tuple[int, ...]:
        """The microbatches of one pass in the order their backward runs."""
        walk = self.positions(pass_index)
        return tuple(reversed(walk)) if self.reverse_breadth else tuple(walk)

    def creates(self, position: int, stage_index: int, *, stage_count: int) -> bool:
        """Whether this microbatch's backward for the stage creates its gradient.

        Only the first pass creates anything, and within it the first
        microbatch the backward walk reaches. The paired last stage walks in
        forward order whatever the flag says, so its creator is the pass's
        first microbatch even when every other stage's is its last.
        """
        if not 0 <= stage_index < stage_count:
            raise IndexError(f"stage {stage_index} of {stage_count}")
        pass_index, slot = divmod(position, self.breadth)
        if pass_index != 0:
            return False
        if self.pair_loss and stage_index == stage_count - 1:
            return slot == 0
        return slot == (self.breadth - 1 if self.reverse_breadth else 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "depth": self.depth,
            "breadth": self.breadth,
            "reverse_breadth": self.reverse_breadth,
            "pair_loss": self.pair_loss,
        }

    @classmethod
    def from_dict(cls, value: object, path: str) -> StepDataOrdering:
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected an object")
        try:
            return cls(
                value["depth"],
                value["breadth"],
                value.get("reverse_breadth", True),
                value.get("pair_loss", True),
            )
        except (KeyError, ValueError) as error:
            raise ValueError(f"{path}: {error}") from error


__all__ = ["StepDataOrdering"]
