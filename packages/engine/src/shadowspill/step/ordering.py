"""How a training step walks its microbatches."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class StepDataOrdering:
    """Which microbatch runs which stage when, within one accumulated step.

    A step is ``depth`` passes over ``breadth`` microbatches each. Within a pass
    every microbatch runs one stage's forward before any runs the next stage's,
    and the backward walks the stages in reverse the same way, so each stage's
    parameters are fetched once per pass rather than once per microbatch. At
    ``breadth == 1`` this is the microbatch-major order, and both flags have no
    effect.

    A stage's gradient is created by the first backward the walk runs for it and
    accumulated into by every later one; `creates` says which that is.

    Parameters
    ----------
    depth : int
        Number of passes. Positive.
    breadth : int
        Microbatches per pass. Positive.
    reverse_breadth : bool, default True
        Whether the backward walks a pass's microbatches in reverse, so the
        freshest activations go first.
    pair_loss : bool, default True
        Whether each microbatch runs its last forward stage and that stage's
        backward together, so the loss's saved state is consumed at once
        instead of held for the whole pass.

    Raises
    ------
    ValueError
        If a count is not a positive int or a flag is not a bool.
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
        """Microbatches in the step: ``depth * breadth``."""
        return self.depth * self.breadth

    @property
    def label(self) -> str:
        """The ordering's name, such as ``"2x4rp"``.

        ``<depth>x<breadth>``, then ``r`` and ``p`` for the flags that are on.
        """
        return (
            f"{self.depth}x{self.breadth}"
            f"{'r' if self.reverse_breadth else ''}{'p' if self.pair_loss else ''}"
        )

    @classmethod
    def from_label(cls, label: str) -> StepDataOrdering:
        """Read an ordering from its `label`.

        Parameters
        ----------
        label : str
            ``<depth>x<breadth>``, then ``r`` and ``p`` for the flags that are on;
            an absent letter turns its flag off.

        Returns
        -------
        StepDataOrdering
            The ordering.

        Raises
        ------
        ValueError
            If ``label`` does not have that form or names a count below one.
        """
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
        """Build the ordering with one microbatch per pass: the microbatch-major order.

        Parameters
        ----------
        microbatches : int
            Microbatches in the step, which is the number of passes.
        reverse_breadth : bool, default True
            As in `StepDataOrdering`.
        pair_loss : bool, default True
            As in `StepDataOrdering`.

        Returns
        -------
        StepDataOrdering
            The ordering ``microbatches x 1``.

        Raises
        ------
        ValueError
            If ``microbatches`` is not a positive int.
        """
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
        """Build the ordering a caller asked for, filling in the count left out.

        Neither count given means depth-first; one given fixes the other; both
        given must multiply to the microbatch count.

        Parameters
        ----------
        microbatches : int
            Microbatches in the step. Positive.
        depth : int, optional
            Number of passes.
        breadth : int, optional
            Microbatches per pass.
        reverse_breadth : bool, default True
            As in `StepDataOrdering`.
        pair_loss : bool, default True
            As in `StepDataOrdering`.

        Returns
        -------
        StepDataOrdering
            The ordering.

        Raises
        ------
        ValueError
            If ``microbatches`` is below one, a given count does not divide it, or
            the two counts do not multiply to it. The message names all three
            numbers.
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
        """Return the microbatches of one pass, in forward order.

        Parameters
        ----------
        pass_index : int
            The pass, from zero.

        Returns
        -------
        range
            The pass's microbatch positions.

        Raises
        ------
        IndexError
            If ``pass_index`` is not below ``depth``.
        """
        if not 0 <= pass_index < self.depth:
            raise IndexError(f"pass {pass_index} of {self.depth}")
        return range(pass_index * self.breadth, (pass_index + 1) * self.breadth)

    def backward_positions(self, pass_index: int) -> tuple[int, ...]:
        """Return the microbatches of one pass in the order their backward runs.

        Parameters
        ----------
        pass_index : int
            The pass, from zero.

        Returns
        -------
        tuple of int
            The pass's microbatch positions, reversed when ``reverse_breadth`` is
            on.

        Raises
        ------
        IndexError
            If ``pass_index`` is not below ``depth``.
        """
        walk = self.positions(pass_index)
        return tuple(reversed(walk)) if self.reverse_breadth else tuple(walk)

    def creates(self, position: int, stage_index: int, *, stage_count: int) -> bool:
        """Return whether a microbatch's backward for a stage creates its gradient.

        Only the first pass creates gradients, and within it the first microbatch
        the backward walk reaches. With ``pair_loss`` the last stage walks in
        forward order whatever ``reverse_breadth`` says, so its creator is the pass's
        first microbatch.

        Parameters
        ----------
        position : int
            The microbatch, from zero across the whole step.
        stage_index : int
            The stage, from zero.
        stage_count : int
            Number of stages.

        Returns
        -------
        bool
            True when this backward creates the gradient, False when it adds into
            it.

        Raises
        ------
        IndexError
            If ``stage_index`` is not below ``stage_count``.
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
        """Return the ordering as JSON values.

        Returns
        -------
        dict
            Its fields by name.
        """
        return {
            "depth": self.depth,
            "breadth": self.breadth,
            "reverse_breadth": self.reverse_breadth,
            "pair_loss": self.pair_loss,
        }

    @classmethod
    def from_dict(cls, value: object, path: str) -> StepDataOrdering:
        """Read an ordering from its `to_dict` form.

        Parameters
        ----------
        value : dict
            The decoded JSON value. ``reverse_breadth`` and ``pair_loss`` default
            to True when absent.
        path : str
            Where the value sits in its document; error messages start with it.

        Returns
        -------
        StepDataOrdering
            The ordering.

        Raises
        ------
        ValueError
            If ``value`` is not a dict, lacks a count, or holds an invalid field.
        """
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
