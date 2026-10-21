"""The settings of one ShadowSpill case, shared by the trainer and the sweep.

On top of what every engine takes (``baselines.common.settings``), the
budgets ShadowSpill plans within: the execution pool on the device and the
spill pool in pinned host memory, with the headrooms the runtime keeps
beside them, the artifact store its builds and plans persist in (the
library's own by default, so every case and every experiment builds each
model once), and the resolution options the planner searches. ShadowSpill
honors the whole precision contract. There is no recompute fraction here: the planner
decides what to keep, recompute or spill, so ``recompute`` is none and any
value is refused; the resolution options are the shares of the flexible
groups it may recompute, of which it plans each and keeps the best.
"""

from __future__ import annotations

import dataclasses
import math

from baselines.common import settings as common
from baselines.common.precision import Capabilities
from baselines.common.settings import GEOMETRY_FIELDS, MODEL_FIELDS, PRECISION_FIELDS

#: The sources whose digests identify this engine in a manifest and a row.
SOURCE_PATHS = (
    "baselines/__init__.py",
    "baselines/common",
    "baselines/engines/shadowspill",
    "workloads/models",
    "workloads/lora",
    "workloads/full_model.py",
    "src/shadowspill",
)

#: The environment beside the launcher's that the trainer runs in.
ENVIRONMENT = "shadowspill"

#: ShadowSpill honors every choice of the contract.
CAPABILITIES = Capabilities("shadowspill")


@dataclasses.dataclass(frozen=True)
class Settings(common.Settings):
    CAPABILITIES: Capabilities = dataclasses.field(
        default=CAPABILITIES, init=False, repr=False, compare=False
    )

    #: The planner decides what to keep, recompute or spill.
    recompute: float | None = None
    # The budgets and the pools.
    execution_gib: float = 16.0
    spill_gib: float = 64.0
    external_headroom_gib: float = 0.5
    host_headroom_gib: float = 2.0
    #: Where builds and plans persist: ``default``, the library's own store;
    #: ``none``, a store of the case's own that ends with it; or a directory.
    artifact_store: str = "default"
    #: The shares of the flexible groups the planner may recompute, as the
    #: quickstart's ``--resolution-options`` spells them.
    resolution_options: str = "quarters"

    def engine_validation(self, request, geometry, precision):
        if self.recompute is not None:
            raise ValueError(
                "ShadowSpill's planner decides what to keep, recompute or spill;"
                " recompute is not a setting here"
            )
        for name in ("execution_gib", "spill_gib"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a positive finite number")
        if not 0 <= self.external_headroom_gib < self.execution_gib:
            raise ValueError(
                "external headroom is nonnegative and smaller than the execution budget"
            )
        if self.host_headroom_gib < 0:
            raise ValueError("host headroom is nonnegative")
        from shadowspill.planner.search.toolkit.resolution import (
            named_resolution_options,
        )

        named_resolution_options(self.resolution_options)
        if not isinstance(self.artifact_store, str) or not self.artifact_store:
            raise ValueError("artifact_store is default, none or a directory")
        if self.allocator_cap_gib is not None:
            raise ValueError(
                "the execution budget is ShadowSpill's device cap; PyTorch's"
                " allocator cap does not exist under its allocator"
            )


HELP = {
    "recompute": "not a setting here: ShadowSpill's planner decides what to keep,"
    " recompute or spill",
    "execution_gib": "the device budget: the execution pool's physical capacity,"
    " external headroom included",
    "spill_gib": "the budget of the spill pool in pinned host memory, held whole"
    " from the start",
    "external_headroom_gib": "device memory reserved inside the execution budget"
    " for allocations outside the pool",
    "host_headroom_gib": "host memory reserved per process beside the pinned spill"
    " pool",
    "artifact_store": "where builds and plans persist, read where they hit and"
    " written where they miss: 'default', the library's own store"
    " (~/.cache/shadowspill); 'none', a store of the case's own that ends with"
    " it; or a directory",
    "resolution_options": "the shares of the flexible groups the planner may"
    " recompute, each planned and the best kept: 'quarters' (its default),"
    " 'eighths', 'halves', or comma-separated exact fractions such as 0,1/2,1",
    "allocator_cap_gib": "not a setting here: the execution budget is the cap",
}


def add_arguments(parser):
    common.add_arguments(
        parser,
        Settings,
        help=HELP,
        types={"recompute": float},
    )


def from_namespace(arguments) -> Settings:
    return common.from_namespace(Settings, arguments)


__all__ = [
    "CAPABILITIES",
    "ENVIRONMENT",
    "GEOMETRY_FIELDS",
    "MODEL_FIELDS",
    "PRECISION_FIELDS",
    "SOURCE_PATHS",
    "Settings",
    "add_arguments",
    "from_namespace",
]
