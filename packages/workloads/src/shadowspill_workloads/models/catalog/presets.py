"""Catalog entries derived from the family-owned declarations."""

from shadowspill_workloads.models.families import families

from .entry import Entry, Geometry


def _entries(family):
    return tuple(
        Entry(
            name=family.name.replace("_", "-")
            + ("" if size == "published" else f"-{size}"),
            family=family.name,
            size=size,
            implementations=family.implementations,
            geometry=Geometry(1024, tokens, 65536),
            mixture=family.mixture,
            expert_parallel=family.expert_parallel,
            description=description,
        )
        for size, tokens, description in family.presets
    )


ENTRIES: tuple[Entry, ...] = tuple(
    item for family in families() for item in _entries(family)
)
