"""The settings of one ShadowSpill case, shared by the trainer and the sweep.

On top of what every engine takes (``baselines.common.settings``), the
budgets ShadowSpill plans within: the execution pool on the device and the
spill pool in pinned host memory, with the headrooms the runtime keeps
beside them, the artifact store its builds and plans persist in (the
library's own by default), the resolution options the planner searches, and the
microbatch search. One case plans every split of the update into
microbatches within ``min_tokens_per_microbatch`` and
``max_tokens_per_microbatch`` (both optional), each under the
``orderings`` it tries, and runs the fastest that fits; a
``tokens_per_microbatch`` pins one size instead. ShadowSpill honors the
whole precision contract. There is no recompute fraction here: the planner
decides what to keep, recompute or spill, so ``recompute`` is none and any
value is refused; the resolution options are the shares of the flexible
groups it may recompute, of which it plans each and keeps the best.

Compatible single-rank sweep cases share explicit preparation artifacts across
execution budgets. The store alone does not bypass objective capture.
"""

from __future__ import annotations

import dataclasses
import math

from baselines.common import settings as common
from baselines.common.geometry import Geometry
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
    "workloads/recipes/text/geometry.py",
    "src/shadowspill",
)

#: The environment beside the launcher's that the trainer runs in.
ENVIRONMENT = "shadowspill"

# Compatible single-rank cases can prepare once and admit separately.
SHARED_PREPARATION = True

#: ShadowSpill honors every choice of the contract.
CAPABILITIES = Capabilities("shadowspill")

#: The microbatch orderings the search tries per geometry: every depth x
#: breadth factor pair, or the depth-first walk alone.
ORDERINGS = ("factors", "depth-first")


@dataclasses.dataclass(frozen=True)
class Settings(common.Settings):
    CAPABILITIES: Capabilities = dataclasses.field(
        default=CAPABILITIES, init=False, repr=False, compare=False
    )

    #: The planner decides what to keep, recompute or spill.
    recompute: float | None = None
    #: One microbatch size, or none: the search plans every split of the
    #: update within the bounds below and keeps the fastest that fits.
    tokens_per_microbatch: int | None = None
    min_tokens_per_microbatch: int | None = None
    max_tokens_per_microbatch: int | None = None
    orderings: str = "factors"
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

    def microbatch_candidates(self, world_size) -> tuple[int, ...]:
        """The tokens per microbatch the search plans, largest first: the
        one a case pins, or every split of a replica's update into whole
        sequences within the bounds."""

        if self.tokens_per_microbatch is not None:
            return (self.tokens_per_microbatch,)
        if self.tokens_per_step is None:
            raise ValueError("a case names its tokens_per_step")
        from workloads.recipes.text.geometry import search_geometries

        replica = self.tokens_per_step // world_size
        if replica % self.sequence_length or self.tokens_per_step % world_size:
            raise ValueError("an update is whole sequences on every rank")
        admitted, _ = search_geometries(
            replica // self.sequence_length,
            sequence_length=self.sequence_length,
            min_tokens_per_microbatch=self.min_tokens_per_microbatch,
            max_tokens_per_microbatch=self.max_tokens_per_microbatch,
        )
        if not admitted:
            raise ValueError("no microbatch size lies within the bounds")
        return tuple(sequences * self.sequence_length for sequences, _ in admitted)

    def geometry(self, world_size) -> Geometry:
        """The case's geometry; while the search is to choose the microbatch,
        it names the largest candidate, and the row names the one chosen."""

        if self.tokens_per_microbatch is not None:
            return super().geometry(world_size)
        return Geometry(
            self.sequence_length,
            self.tokens_per_step,
            self.microbatch_candidates(world_size)[0],
            world_size,
        )

    def engine_validation(self, request, geometry, precision):
        for name in (
            "tokens_per_microbatch",
            "min_tokens_per_microbatch",
            "max_tokens_per_microbatch",
        ):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError(f"{name} is a positive integer or none")
        if self.tokens_per_microbatch is not None and (
            self.min_tokens_per_microbatch is not None
            or self.max_tokens_per_microbatch is not None
        ):
            raise ValueError(
                "a pinned tokens_per_microbatch leaves nothing for bounds to bound"
            )
        if self.orderings not in ORDERINGS:
            raise ValueError(f"orderings must be one of {ORDERINGS}")
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
    "min_tokens_per_microbatch": "the smallest microbatch the search plans, in"
    " tokens; none for one sequence",
    "max_tokens_per_microbatch": "the largest microbatch the search plans, in"
    " tokens; none for a replica's whole update",
    "orderings": "the microbatch orderings the search tries per geometry:"
    " 'factors', every depth x breadth pair (the default), or 'depth-first'",
}


def add_arguments(parser):
    common.add_arguments(
        parser,
        Settings,
        help=HELP,
        types={
            "recompute": float,
            "min_tokens_per_microbatch": int,
            "max_tokens_per_microbatch": int,
        },
        choices={"orderings": ORDERINGS},
    )


def from_namespace(arguments) -> Settings:
    return common.from_namespace(Settings, arguments)


__all__ = [
    "CAPABILITIES",
    "ENVIRONMENT",
    "GEOMETRY_FIELDS",
    "MODEL_FIELDS",
    "ORDERINGS",
    "PRECISION_FIELDS",
    "SOURCE_PATHS",
    "Settings",
    "add_arguments",
    "from_namespace",
]
