"""Build-only setup returned by an adapter; no live setup enters saved Plans."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BuildSetup:
    model: Any
    optimizer: Any
    objective: Any
    build_options: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
