"""Readable capture configurations from the same resolved preparation inventory."""

import json
from collections import defaultdict

from shadowspill.diagnostics.progress import event
from shadowspill.training.preparation.capture.templates import (
    CaptureTemplate,
    TemplateInventory,
)
from shadowspill_client import Policy


def candidate_title(template: CaptureTemplate, index: int, count: int) -> str:
    purpose = template.kind.replace("_", " ").capitalize()
    return f"Candidate {index}/{count} · {purpose} · {template.id[:12]}"


def describe_candidate(inventory: TemplateInventory, template: CaptureTemplate) -> None:
    sizes = {
        inventory.layouts.setup.config.preparation.cases[
            row.layout.candidate.case
        ].microbatch_size
        for rows in inventory.families.values()
        for row in rows
        if any(
            use is not None and use.template == template.id
            for worker in row.workers.values()
            for use in (worker.main, worker.optimizer)
        )
    }
    buffer = template.binding.inputs.layout.buffer
    event("Microbatch size: " + ", ".join(f"{size:,}" for size in sorted(sizes)))
    event(f"Maximum sequences/items per microbatch: {buffer.max_datums:,}")
    limits = template.binding.inputs.layout.limits
    if "max_sequence_length" in limits:
        event(f"Maximum sequence length: {limits['max_sequence_length']:,}")
    fields: dict[tuple[int, ...], list[str]] = defaultdict(list)
    for path, shape in buffer.fields.items():
        fields[tuple(shape)].append(".".join(path))
    for shape, names in fields.items():
        dimensions = " x ".join(f"{size:,}" for size in shape) or "Scalar"
        event(f"Buffer [{dimensions}]: {', '.join(names)}")
    defaults = {
        "partition": "Automatic",
        "chunking": "Disabled",
        "graph_pair_policy": "Cost-input (default)",
    }
    titles = {
        "partition": "Stage partition",
        "chunking": "Execution chunking",
        "graph_pair_policy": "Retention",
    }
    for name, value in template.build.to_dict().items():
        if value is None:
            description = defaults[name]
        else:
            policy = Policy.from_dict(value)
            description = policy.name
            if policy.config:
                description += " " + json.dumps(
                    policy.to_dict()["config"], sort_keys=True
                )
        event(f"{titles[name]}: {description}")
    if template.kind != "forward":
        event(
            "Terminal stage: Saved; ordering search: All supported, paired by default"
        )
