"""The two origins a task storage can have, as the capture paths name them."""

from __future__ import annotations

from dataclasses import dataclass

from torch.fx import Node


@dataclass(frozen=True, slots=True)
class _InputRoot:
    position: int


@dataclass(frozen=True, slots=True)
class _FreshRoot:
    node: Node
    result_index: int


_SemanticRoot = _InputRoot | _FreshRoot
