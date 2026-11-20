"""Total per-rank GPU allowance and the separate spill capacity."""

from __future__ import annotations

from dataclasses import dataclass

from shadowspill.ir.validation import expect_integer, expect_mapping, field

from ._values import nonnegative_int


@dataclass(frozen=True)
class MemoryBudgets:
    """Memory budgets for one worker, in bytes.

    Parameters
    ----------
    execution_bytes : int
        Total device memory the worker may use, in bytes. Positive. It covers
        the fixed device overhead as well as the execution pool.
    spill_bytes : int
        Capacity of the worker's spill pool, in bytes. At least zero.

    Raises
    ------
    ValueError
        If a budget is not a nonnegative integer, or ``execution_bytes`` is
        zero.
    """

    execution_bytes: int
    spill_bytes: int

    def __post_init__(self) -> None:
        nonnegative_int(self.execution_bytes, "execution_bytes")
        nonnegative_int(self.spill_bytes, "spill_bytes")
        if not self.execution_bytes:
            raise ValueError("execution_bytes must be positive")

    def to_dict(self) -> dict[str, int]:
        """Return the budgets as plain data.

        Returns
        -------
        dict
            ``execution_bytes`` and ``spill_bytes``.
        """
        return {
            "execution_bytes": self.execution_bytes,
            "spill_bytes": self.spill_bytes,
        }

    @classmethod
    def from_value(cls, value: object, path: str = "memory_budgets") -> MemoryBudgets:
        """Read budgets from a `to_dict` record.

        Parameters
        ----------
        value : object
            A mapping with integer ``execution_bytes`` and ``spill_bytes``.
        path : str, default "memory_budgets"
            Where ``value`` came from, named in error messages.

        Returns
        -------
        MemoryBudgets
            The budgets.

        Raises
        ------
        ValueError
            If ``value`` is not such a mapping, or a budget is out of range.
        """
        data = expect_mapping(value, path)
        return cls(
            expect_integer(
                field(data, "execution_bytes", path), f"{path}.execution_bytes"
            ),
            expect_integer(field(data, "spill_bytes", path), f"{path}.spill_bytes"),
        )
