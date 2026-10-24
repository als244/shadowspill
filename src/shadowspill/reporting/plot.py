"""Standalone timeline and memory figures; imports plotting only on request."""

from pathlib import Path

from shadowspill.artifacts.plan import Plan
from shadowspill.ir import ResourceKind

from .rows import transfer_rows


def plot_plan(plan: Plan, path: Path) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(12, 6), layout="constrained")
    FigureCanvasAgg(figure)
    timeline, memory = figure.subplots(2, 1)
    lanes = {"compute": 2, "fetch": 1, "evict": 0}
    has_collectives = any(
        task.resource_kind is ResourceKind.COMMUNICATION
        for task in plan.timeline.task_intervals
    )
    for task in plan.timeline.task_intervals:
        collective = task.resource_kind is ResourceKind.COMMUNICATION
        timeline.broken_barh(
            [(task.start_ns / 1e6, (task.end_ns - task.start_ns) / 1e6)],
            (2.7 if collective else 1.7, 0.6),
            facecolors="#8c5fa8" if collective else "#526ab2",
        )
    for transfer in transfer_rows(plan):
        timeline.broken_barh(
            [(transfer["start_seconds"] * 1e3, transfer["duration_seconds"] * 1e3)],
            (lanes[transfer["direction"]] - 0.3, 0.6),
            facecolors="#8c5fa8"
            if transfer["kind"] == "copy_and_collective"
            else "#c87822"
            if transfer["direction"] == "fetch"
            else "#348573",
        )
    labels = ["Evict transfer", "Fetch transfer", "Compute"]
    if has_collectives:
        labels.append("Collective tasks")
    timeline.set_yticks(range(len(labels)), labels)
    timeline.set_title(f"{plan.program.name or 'Program'} — simulated timeline")
    timeline.set_xlabel("Time (ms)")
    if any(row["kind"] == "copy_and_collective" for row in transfer_rows(plan)):
        timeline.set_xlabel("Time (ms) — purple: combined host copy and collective")
    timeline.grid(axis="x", alpha=0.2)
    points = plan.timeline.memory_timeline
    times = [item.time_ns / 1e6 for item in points]
    device = plan.program.devices[0].device_id
    overhead = plan.fixed_device_bytes
    if points:
        gpu = [
            dict(item.device_object_bytes)[device]
            + dict(item.device_workspace_bytes)[device]
            + overhead
            for item in points
        ]
        spill = [item.spill_bytes for item in points]
        memory.step(
            times, [item / 2**30 for item in gpu], where="post", label="GPU total"
        )
        memory.step(
            times, [item / 2**30 for item in spill], where="post", label="Spill"
        )
        memory.axhline(
            plan.memory_budgets.execution_bytes / 2**30,
            color="#526ab2",
            linestyle="--",
            label="GPU allowance",
        )
        memory.axhline(
            plan.memory_budgets.spill_bytes / 2**30,
            color="#348573",
            linestyle=":",
            label="Spill capacity",
        )
        memory.set_xlabel("Time (ms)")
    else:
        # Native simulation currently records peaks, not an allocation event
        # history. Do not invent a memory curve from task/transfer intervals.
        memory.bar(
            [0, 1],
            [
                plan.memory_budgets.execution_bytes / 2**30,
                plan.memory_budgets.spill_bytes / 2**30,
            ],
            width=0.6,
            color="#e1e6ee",
            label="Capacity",
        )
        memory.bar(
            [0, 1],
            [
                plan.summary["execution_peak_bytes"] / 2**30,
                plan.summary["spill_peak_bytes"] / 2**30,
            ],
            width=0.35,
            color=["#526ab2", "#348573"],
            label="Simulated peak",
        )
        memory.set_xticks([0, 1], ["GPU including fixed overhead", "Spill"])
        memory.set_title("Memory peaks — allocation history unavailable")
    memory.set_ylabel("Memory (GiB)")
    memory.legend(loc="upper right", ncols=2)
    memory.grid(alpha=0.2)
    figure.savefig(path / "timeline.svg")
    figure.savefig(path / "timeline.png", dpi=150)
    figure.clear()
