"""Architecture-independent controls for bounded axis chunking."""

from .geometry import ChunkAxis
from .lifetime_policy import (
    ExecutionLifetimes,
    LayoutIndependentOperator,
    LifetimeOperation,
    LifetimeRule,
    LifetimeSelection,
    ReadonlyOperator,
)
from .policy import (
    AxisChunks,
    AxisFrom,
    ChunkLocal,
    ChunkPolicy,
    ChunkRegions,
    OperatorRule,
    StageInfo,
)

__all__ = [
    "AxisChunks",
    "AxisFrom",
    "ChunkAxis",
    "ChunkLocal",
    "ChunkPolicy",
    "ChunkRegions",
    "ExecutionLifetimes",
    "LayoutIndependentOperator",
    "LifetimeOperation",
    "LifetimeRule",
    "LifetimeSelection",
    "OperatorRule",
    "ReadonlyOperator",
    "StageInfo",
]
