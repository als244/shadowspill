"""Borrowed group identities for model copying and captured collective calls.

PyTorch group names are process-local counters. Artifact identities instead use
caller-supplied logical names plus ordered rank membership. Private c10d calls
are isolated here; no process group is constructed or destroyed by this module.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Self, cast

import torch.distributed as dist
from torch._C._distributed_c10d import (
    _register_process_group,
    _unregister_process_group,
)
from torch.fx import GraphModule

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import GroupName


_ACTIVE: dict[str, tuple[dist.ProcessGroup, int]] = {}


class Bindings:
    def __init__(self, groups: Mapping[str, dist.ProcessGroup]) -> None:
        self.groups = dict(groups)
        self.aliases: dict[str, str] = {}
        self.source_names: dict[str, str] = {}
        self._owned: list[str] = []
        try:
            for logical, group in sorted(self.groups.items()):
                if not logical or not isinstance(logical, str):
                    raise ValueError(
                        "communication group names must be nonempty strings"
                    )
                members = tuple(dist.get_process_group_ranks(group))
                if dist.get_rank() not in members:
                    raise ValueError(f"this process is not a member of {logical}")
                identity = json.dumps([logical, members], separators=(",", ":"))
                alias = (
                    "shadowspill/group/" + hashlib.sha256(identity.encode()).hexdigest()
                )
                held = _ACTIVE.get(alias)
                if held is not None and held[0] is not group:
                    raise ValueError(
                        "another live process group is already bound to "
                        f"logical name {logical!r}"
                    )
                if held is None:
                    _register_process_group(cast("GroupName", alias), group)
                    _ACTIVE[alias] = (group, 1)
                else:
                    _ACTIVE[alias] = (group, held[1] + 1)
                self._owned.append(alias)
                self.aliases[logical] = alias
                self.source_names.setdefault(group.group_name, alias)
        except BaseException:
            self.close()
            raise

    def copy_memo(self) -> dict[int, Any]:
        return {id(group): group for group in self.groups.values()}

    def rewrite_collectives(self, graph_module: GraphModule) -> GraphModule:
        """Rewrite only the schema's explicit group_name argument."""
        for node in graph_module.graph.nodes:
            schema = getattr(node.target, "_schema", None)
            if schema is None or not schema.name.startswith("_c10d_functional::"):
                continue
            arguments = list(node.args)
            keywords = dict(node.kwargs)
            for position, argument in enumerate(schema.arguments):
                if argument.name != "group_name":
                    continue
                name = (
                    arguments[position]
                    if position < len(arguments)
                    else keywords[argument.name]
                )
                if not isinstance(name, str):
                    raise ValueError(
                        "a collective group must be an explicit named binding"
                    )
                if name in self.aliases.values():
                    continue
                if name not in self.source_names:
                    raise ValueError(
                        f"collective names an undeclared process group {name!r}"
                    )
                if position < len(arguments):
                    arguments[position] = self.source_names[name]
                else:
                    keywords[argument.name] = self.source_names[name]
            node.args, node.kwargs = tuple(arguments), keywords
        graph_module.recompile()
        return graph_module

    def close(self) -> None:
        for alias in reversed(self._owned):
            group, count = _ACTIVE[alias]
            if count == 1:
                _unregister_process_group(cast("GroupName", alias))
                del _ACTIVE[alias]
            else:
                _ACTIVE[alias] = group, count - 1
        self._owned.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *error: object) -> None:
        self.close()
