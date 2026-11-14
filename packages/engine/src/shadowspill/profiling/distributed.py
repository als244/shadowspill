"""Conservative symmetric profiles when model tasks themselves communicate."""

from dataclasses import replace

from shadowspill.artifacts._values import plain
from shadowspill.artifacts.profiles import Profiles
from shadowspill.communication.binding.groups import GroupBindings


def combine_profiles(local: Profiles, groups: GroupBindings) -> Profiles:
    rows = [Profiles.from_dict(row) for row in groups.gather(local.to_dict())]
    for row in rows:
        if row.setup != local.setup or set(row.tasks) != set(local.tasks):
            raise ValueError("model-parallel profiling requires symmetric task forms")
    for key, current in tuple(local.tasks.items()):
        records = [row.tasks[key] for row in rows]
        if any(
            r.duration_seconds is None or r.workspace_bytes is None for r in records
        ):
            raise ValueError(f"task {key}: workers returned incomplete measurements")
        # A shared executable Plan needs the same physical task contract. Larger
        # observed overheads/times can be conservatively combined; different
        # allocation sequences or layouts cannot.
        for field in (
            "allocation_contract",
            "compiled_layout",
            "output_input_bindings",
        ):
            if any(
                record.metadata.get(field) != current.metadata.get(field)
                for record in records
            ):
                raise ValueError(f"task {key}: workers observed different {field}")
        if any(r.compatibility_digest != current.compatibility_digest for r in records):
            raise ValueError(f"task {key}: workers built different implementations")
        local.tasks[key] = replace(
            current,
            duration_seconds=max(r.duration_seconds or 0.0 for r in records),
            workspace_bytes=max(r.workspace_bytes or 0 for r in records),
            metadata={
                **plain(current.metadata),
                "rank_duration_seconds": [r.duration_seconds for r in records],
            },
        )
    if any(row.fixed_device_bytes is None for row in rows):
        local.fixed_device_bytes = None
    else:
        local.fixed_device_bytes = max(row.fixed_device_bytes or 0 for row in rows)
    local.metadata = {
        **local.metadata,
        "rank_aggregation": "maximum; identical allocation contracts",
    }
    return local
