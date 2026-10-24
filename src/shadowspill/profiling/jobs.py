"""Enumerate unique compiled forms while retaining forward/backward pairing."""

from collections.abc import Iterator, Sequence
from typing import Any

from shadowspill.artifacts.program import Program


def stage_jobs(
    program: Program, stages: Sequence[int] | None
) -> Iterator[dict[str, Any]]:
    templates = program.bindings.get("training_templates")
    if templates is not None:
        inventory = templates[0]
        selected = _selection(stages, len(inventory))
        for index, forms in enumerate(inventory):
            if index in selected:
                for form in forms:
                    yield {
                        "stage": index,
                        "forward": form["forward"],
                        "backward": form["backward"],
                    }
        if stages is None:
            for component in program.bindings.get("optimizer_templates", ()):
                yield {"optimizer": component}
            for completion in program.bindings.get("gradient_templates", ()):
                if completion["padding"] is not None:
                    yield {"gradient": completion["padding"]}
                yield {"gradient": {"profile_key": completion["accumulation_key"]}}
    else:
        tasks = {task.task_id: task for task in program.tasks}
        entries = program.bindings["entrypoints"]
        selected = _selection(stages, len(entries))
        for index, entry in enumerate(entries):
            if index in selected:
                yield {
                    "stage": index,
                    "forward": {
                        "profile_key": tasks[entry["task_id"]].profile_id,
                        "inputs": entry["input_slots"],
                        "outputs": entry["output_slots"],
                    },
                }


def _selection(stages: Sequence[int] | None, count: int) -> set[int]:
    selected = set(range(count)) if stages is None else set(stages)
    if any(
        not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < count
        for index in selected
    ):
        raise ValueError(f"stages must contain indices from 0 through {count - 1}")
    return selected
