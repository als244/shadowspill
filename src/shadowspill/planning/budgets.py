"""Total per-rank GPU allowance and the separate spill capacity."""

from __future__ import annotations

from dataclasses import dataclass

from shadowspill.artifacts._values import nonnegative_int
from shadowspill.ir.validation import expect_integer, expect_mapping, field


@dataclass(frozen=True)
class MemoryBudgets:
    execution_bytes: int
    spill_bytes: int

    def __post_init__(self) -> None:
        nonnegative_int(self.execution_bytes, "execution_bytes")
        nonnegative_int(self.spill_bytes, "spill_bytes")
        if not self.execution_bytes:
            raise ValueError("execution_bytes must be positive")

    def to_dict(self) -> dict[str, int]:
        return {
            "execution_bytes": self.execution_bytes,
            "spill_bytes": self.spill_bytes,
        }

    @classmethod
    def from_value(cls, value: object, path: str = "memory_budgets") -> MemoryBudgets:
        data = expect_mapping(value, path)
        return cls(
            expect_integer(
                field(data, "execution_bytes", path), f"{path}.execution_bytes"
            ),
            expect_integer(field(data, "spill_bytes", path), f"{path}.spill_bytes"),
        )
