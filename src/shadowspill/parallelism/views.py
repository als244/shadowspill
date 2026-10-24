"""Resolve saved owned-parameter views without a communicator or device."""

from typing import TYPE_CHECKING

from .groups import GroupRef

if TYPE_CHECKING:
    from shadowspill.artifacts.program import Program


def owned_view_offsets(program: "Program", logical_rank: int) -> dict[str, int]:
    """Return dtype-element offsets into each view's full compute allocation."""
    world = 1 if program.parallelism is None else program.parallelism.world_size
    if type(logical_rank) is not int or not 0 <= logical_rank < world:
        raise ValueError("logical_rank is outside the saved Program")
    objects = {item.object_id: item for item in program.objects}
    aliases = {item.alias_group_id: item for item in program.alias_groups}
    result = {}
    for value, description in program.bindings.get("owned_parameter_views", {}).items():
        if value not in objects:
            raise ValueError(f"owned parameter view names an unknown object: {value}")
        group = GroupRef(value, description["groups"])
        if sorted(rank for members in group.groups for rank in members) != list(
            range(world)
        ):
            raise ValueError(f"{value}: owned groups must cover every logical rank")
        elements = description["elements"]
        view = objects[value]
        full = aliases[view.alias_group_id]
        if (
            type(elements) is not int
            or elements <= 0
            or view.offset_bytes != 0
            or view.size_bytes % elements
            or view.size_bytes * group.size != full.size_bytes
        ):
            raise ValueError(f"{value}: owned view disagrees with full-buffer geometry")
        result[value] = group.index(logical_rank) * elements
    return result
