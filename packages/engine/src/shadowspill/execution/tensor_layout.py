"""Persistent tensor descriptions, without opening runtime or spill storage."""

from collections.abc import Iterable

from shadowspill.artifacts.plan import Plan
from shadowspill.artifacts.state_inventory import program_state
from shadowspill.artifacts.tensors import tensor_geometry

from .tensor_regions import tensor_regions


class TensorLayout:
    def __init__(
        self,
        plan: Plan,
        *,
        logical_rank: int,
        shared_aliases: Iterable[str] = (),
    ) -> None:
        self.plan = plan
        self.spill_sizes = {
            alias.alias_group_id: alias.size_bytes
            if alias.spill_size_bytes is None
            else alias.spill_size_bytes
            for alias in plan._costed_program.alias_groups
        }
        self.regions = tensor_regions(plan.program, logical_rank)
        self.persistent_objects = frozenset(
            binding.object_id
            for binding in program_state(plan.program, logical_rank).bindings
            if binding.object_id is not None
        )
        self.aliases = {o.object_id: o.alias_group_id for o in plan.program.objects}
        self.sizes = {a.alias_group_id: a.size_bytes for a in plan.program.alias_groups}
        self.shared_aliases = frozenset(shared_aliases)
        self.geometry = tensor_geometry(plan.program)
