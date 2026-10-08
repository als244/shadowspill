"""Explicit process groups and parameter replication for coordinated preparation.

Models keep ordinary PyTorch process groups. This binding describes which
parameters are replicas and which gradients still require a SUM; it does not
create groups, place models or choose a loss normalization.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch.distributed as dist
from torch import nn

from ._control import Control
from ._groups import Bindings
from ._ownership import Parameter, members, resolve, synchronize_initial, validate

_INHERIT = object()
_CURRENT: ContextVar[BoundDistributed | None] = ContextVar(
    "shadowspill_preparation", default=None
)


class Distributed:
    """Borrow existing communication groups; coordinate preparation, not execution.

    ``group`` contains every process participating in this workload. Parameters
    default to replicas on that group. Override an iterable of registered
    parameter objects/names with another group, or ``None`` for unique local
    state. Remaining gradient contributions default to each replica group and
    are summed without division; explicit gradient overrides describe reductions
    already performed by a model. Ordinary buffers stay rank-local.

    ``groups`` gives stable names to additional process groups used by the model.
    Each model collective must finish before its task completes. Runtime tasks
    and transfers do not enter the preparation control channel.

    ``symmetric_planning=True`` verifies matching planning memory contracts,
    shares conservative timing estimates and divides CPU search work among
    participants. Mismatches fall back to independent local searches. Each
    rank still captures, profiles, and physically admits its own executable.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        *,
        replica_group: Any = _INHERIT,
        groups: Mapping[str, dist.ProcessGroup] | None = None,
        replica_overrides: Iterable[
            tuple[Iterable[nn.Parameter | str], dist.ProcessGroup | None]
        ] = (),
        gradient_group: Any = _INHERIT,
        gradient_overrides: Iterable[
            tuple[Iterable[nn.Parameter | str], dist.ProcessGroup | None]
        ] = (),
        sync_initial_state: bool = True,
        symmetric_planning: bool = False,
        timeout: float | None = None,
    ) -> None:
        self.group = group
        self.replica_group = group if replica_group is _INHERIT else replica_group
        self.gradient_group = gradient_group
        self.groups = dict(groups or {})
        self.replica_overrides = tuple(
            (tuple(parameters), group) for parameters, group in replica_overrides
        )
        self.gradient_overrides = tuple(
            (tuple(parameters), group) for parameters, group in gradient_overrides
        )
        self.sync_initial_state = sync_initial_state
        self.symmetric_planning = symmetric_planning
        self.timeout = 1800.0 if timeout is None else timeout
        if self.timeout <= 0:
            raise ValueError("distributed preparation timeout must be positive")
        if not isinstance(sync_initial_state, bool):
            raise TypeError("sync_initial_state must be a bool")
        if not isinstance(symmetric_planning, bool):
            raise TypeError("symmetric_planning must be a bool")

    def _bind(
        self, model: nn.Module, control_group: dist.ProcessGroup, *, namespace: str
    ) -> BoundDistributed:
        if str(dist.get_backend(control_group)).lower() != "gloo":
            raise ValueError(
                "distributed preparation requires a caller-owned Gloo control group"
            )
        control = Control(control_group, namespace=namespace, timeout=self.timeout)

        def describe() -> tuple[
            tuple[Parameter, ...],
            dict[str, tuple[dist.ProcessGroup | None, dist.ProcessGroup | None]],
        ]:
            if members(self.group) != control.members:
                raise ValueError(
                    "control and workload groups must have the "
                    "same ordered participants"
                )
            options: dict[str, Any] = dict(
                replica_group=self.replica_group,
                replica_overrides=self.replica_overrides,
                gradient_overrides=self.gradient_overrides,
            )
            if self.gradient_group is not _INHERIT:
                options["gradient_default"] = self.gradient_group
            return resolve(model, **options)

        records, parameter_groups = control.run("ownership/describe", describe)
        validate(control, records)

        def bind_groups() -> Bindings:
            declared = dict(self.groups)
            existing = {id(group) for group in declared.values()}
            automatic = [("participants", self.group), ("replicas", self.replica_group)]
            if self.gradient_group is not _INHERIT:
                automatic.append(("gradients", self.gradient_group))
            automatic.extend(
                (f"replica_override/{index}", group)
                for index, (_, group) in enumerate(self.replica_overrides)
            )
            automatic.extend(
                (f"gradient_override/{index}", group)
                for index, (_, group) in enumerate(self.gradient_overrides)
            )
            for name, group in automatic:
                if group is not None and id(group) not in existing:
                    if name in declared:
                        raise ValueError(
                            f"group name {name!r} is reserved by the default binding"
                        )
                    declared[name] = group
                    existing.add(id(group))
            return Bindings(declared)

        bindings = control.run("groups/bind", bind_groups)
        return BoundDistributed(self, control, records, parameter_groups, bindings)


@dataclass
class BoundDistributed:
    specification: Distributed
    control: Control
    parameters: tuple[Parameter, ...]
    parameter_groups: Mapping[
        str, tuple[dist.ProcessGroup | None, dist.ProcessGroup | None]
    ]
    groups: Bindings
    initialized: bool = False
    shard_optimizer: bool = True

    @contextmanager
    def activate(self) -> Iterator[BoundDistributed]:
        token = _CURRENT.set(self)
        try:
            yield self
        except BaseException as error:
            self.control.fail("prepare", error)
            raise
        finally:
            _CURRENT.reset(token)

    def synchronize_initial(self, model: nn.Module) -> None:
        if self.specification.sync_initial_state:
            synchronize_initial(model, self.parameters, self.control)

    def checkpoint_layout(self) -> dict[str, Any]:
        return {
            "members": list(self.control.members),
            "rank": self.control.rank,
            "parameters": [value.record() for value in self.parameters],
            "groups": self.groups.aliases,
            "shard_optimizer": self.shard_optimizer,
        }

    def close(self) -> None:
        self.groups.close()


def current() -> BoundDistributed | None:
    return _CURRENT.get()


def borrowed_group_memo() -> dict[int, Any]:
    bound = current()
    return {} if bound is None else bound.groups.copy_memo()


__all__ = ["Distributed"]
