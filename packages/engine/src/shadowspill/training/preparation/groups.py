"""Coordinate logical-rank captures without assuming a parallelism layout."""

from .templates import TemplateInventory


def capture_groups(inventory: TemplateInventory) -> tuple[tuple[str, ...], ...]:
    """Order each rank's exact template together with its cooperating peers.

    Template hashes include rank ownership, so sorting hashes independently on
    each worker would not preserve collective participation. Walk the declared
    operation/candidate dependencies once; compatible counts share a group.
    """
    ranks = range(inventory.layouts.setup.context.parallelism.world_size)
    groups: dict[tuple[str, ...], None] = {}
    for candidates in inventory.families.values():
        for candidate in candidates:
            if candidate.workers.keys() != set(ranks):
                raise ValueError("capture candidate must name every logical worker")
            workers = [candidate.workers[rank] for rank in ranks]
            for uses in (
                [worker.main for worker in workers],
                [worker.masked for worker in workers],
            ):
                if all(use is None for use in uses):
                    continue
                if any(use is None for use in uses):
                    raise ValueError("workers disagree on required capture paths")
                keys = tuple(use.template for use in uses if use is not None)
                if any(
                    inventory.templates[key].binding.logical_rank != rank
                    for rank, key in enumerate(keys)
                ):
                    raise ValueError(
                        "capture group differs from logical rank ownership"
                    )
                groups[keys] = None
    if {key for group in groups for key in group} != inventory.templates.keys():
        raise ValueError("capture groups must cover exactly the prepared templates")
    return tuple(groups)
