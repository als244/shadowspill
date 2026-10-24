"""CPU-only collective probe descriptions derived from built transfer geometry."""

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from shadowspill.artifacts.program import Program

_WIDTHS = {
    "torch.uint8": 1,
    "torch.int8": 1,
    "torch.int32": 4,
    "torch.int64": 8,
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.float64": 8,
}


@dataclass(frozen=True, order=True)
class CollectiveRequest:
    operation: str
    ranks: tuple[int, ...]
    tensor_bytes: int
    dtype: str

    def __post_init__(self) -> None:
        if self.operation not in {"all_gather", "reduce_scatter", "all_reduce"}:
            raise ValueError(f"unsupported collective operation: {self.operation}")
        if (
            not self.ranks
            or any(type(rank) is not int or rank < 0 for rank in self.ranks)
            or len(set(self.ranks)) != len(self.ranks)
        ):
            raise ValueError(
                "collective ranks must be ordered, unique nonnegative integers"
            )
        if type(self.tensor_bytes) is not int or self.tensor_bytes <= 0:
            raise ValueError("tensor_bytes must be a positive integer")
        if self.dtype not in _WIDTHS:
            raise ValueError(f"unsupported collective dtype: {self.dtype}")
        divisor = _WIDTHS[self.dtype]
        if self.operation != "all_reduce":
            divisor *= len(self.ranks)
        if self.tensor_bytes % divisor:
            raise ValueError(
                "collective size must contain whole elements in every shard"
            )

    @property
    def elements(self) -> int:
        return self.tensor_bytes // _WIDTHS[self.dtype]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def collective_requests(
    program: "Program | None",
    overrides: Sequence[Mapping[str, Any]] | None,
) -> tuple[CollectiveRequest, ...]:
    """Deduplicate probes, retaining order within each logical participant group.

    Explicit overrides replace derivation; an empty list requests no collectives.
    The initial distributed lowering exposes weight gathers. Training and
    model-owned communication need their own lowered descriptors first.
    """
    rows: list[Mapping[str, Any]] = list(overrides) if overrides is not None else []
    if overrides is None and program is not None:
        if (
            program.parallelism is not None
            and program.parallelism.world_size > 1
            and program.bindings.get("kind") != "forward"
        ):
            raise ValueError(
                "this Program has no supported distributed training communication "
                "lowering; supply explicit collective probes"
            )
        for shard in program.bindings.get("weight_shards", {}).values():
            rows.extend(
                dict(
                    operation="all_gather",
                    ranks=group,
                    tensor_bytes=shard["padded_bytes"],
                    dtype=shard["dtype"],
                )
                for group in shard["groups"]
            )
        from .task_costs import task_collectives

        rows.extend(
            request.to_dict()
            for requests in task_collectives(program).values()
            for request in requests
        )
    requests = {
        CollectiveRequest(**{**row, "ranks": tuple(row["ranks"])}) for row in rows
    }
    return tuple(sorted(requests))
