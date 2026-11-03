"""Architecture-independent controls for bounded axis chunking."""

from .geometry import ChunkAxis
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
    "OperatorRule",
    "StageInfo",
]
