"""Task stalls, compute/fetch/evict execution, and byte-based backlog."""

from __future__ import annotations

import numpy as np
from matplotlib.patches import Rectangle

from ..data import GIB, byte_label
from ..style import BG, GREEN, MUTED, RED, WHITE, duration_label
from .render_common import LANE_COLOR, ROLE_COLOR, PlanCanvas, truncate


class ExecutionRenderer(PlanCanvas):
    """Compose task-centered wait explanations and simulated execution."""

    def draw_scene(self, state):
        if not state["kind"].startswith("execution"):
            raise ValueError("expected an execution scene")
        self.execution(state)

    def execution(self, state):
        if state["kind"] == "execution-sweep" and "task" not in state:
            state = {
                **state,
                "task": min(
                    len(self.data.tasks) - 1,
                    int(
                        np.searchsorted(
                            self.task_ends, state["cursor_ns"], side="right"
                        )
                    ),
                ),
            }
        self.minimap(state)
        lo, hi = state["window_ns"]
        lanes = self.axis(
            (0.07, 0.52, 0.63, 0.235), xlim=(lo / 1e9, hi / 1e9), ylim=(-0.05, 3.45)
        )
        self.lanes(
            lanes, [lo, hi], task_index=state.get("task"), cursor=state["cursor_ns"]
        )
        lanes.set_xticklabels([])
        if "task" in state:
            self.text(
                0.07,
                0.494,
                "Outlined: this task's blockers     Dimmed: other transfers",
                size=10,
                color=MUTED,
            )
        # Queue charts preserve every event; no sampled peak can disappear.
        series = self.data.pending_bytes
        i = max(0, int(np.searchsorted(series.times, lo, side="right")) - 1)
        j = min(
            len(series.times), int(np.searchsorted(series.times, hi, side="right")) + 1
        )
        ceiling = max(1 / 1024, float(series.values[i:j].max(initial=0)) / GIB) * 1.2
        queue = self.axis(
            (0.07, 0.36, 0.63, 0.115),
            xlim=(lo / 1e9, hi / 1e9),
            ylim=(0, ceiling),
            ylabel="Queued host GiB",
        )
        for col, direction in enumerate(series.names):
            queue.step(
                series.times[i:j] / 1e9,
                series.values[i:j, col] / GIB,
                where="post",
                color=LANE_COLOR[direction],
                linewidth=1.4,
                label=direction.title(),
            )
        if series.names:
            queue.legend(
                loc="upper right", frameon=False, labelcolor=WHITE, fontsize=10, ncols=2
            )
        queue.axvline(state["cursor_ns"] / 1e9, color=WHITE, linewidth=0.9)
        queue.set_xticklabels([])
        self.pressure(state, rect=(0.07, 0.125, 0.285, 0.17))
        self.pressure(state, rect=(0.415, 0.125, 0.285, 0.17), spill=True)
        side = self.card(top=0.78)
        self.queue_card(side, state)

    def queue_card(self, side, state):
        if "task" in state:
            self.task_wait_card(side, state)
            return
        instant = state["cursor_ns"]
        y = self.card_text(side, 0.965, f"t = {instant / 1e9:.4f} s", size=18)
        y = self.queue_bars(side, y, instant)
        y = self.card_text(
            side,
            y - 0.035,
            "Backlog counts CPU transfer bytes waiting for the transfer to start.",
            size=12,
            color=MUTED,
        )
        self.card_text(
            side,
            y - 0.025,
            "Solid bars: transfer in progress\nRed hatch: compute waiting",
            size=12,
        )

    def queue_bars(self, side, y, instant, *, compact=False):
        step = 0.105 if compact else 0.145
        bar_y = 0.061 if compact else 0.085
        label_size = 11 if compact else 13
        scale = max(1, int(self.data.pending_bytes.values.max(initial=0)))
        for direction in ("fetch", "evict"):
            pending = self.data.queue(direction, instant)
            total = sum(a["transfer_bytes"] for a in pending)
            side.text(
                0.065,
                y,
                f"{direction.title()} backlog",
                transform=side.transAxes,
                va="top",
                fontsize=label_size,
                color=LANE_COLOR[direction],
            )
            side.text(
                0.935,
                y,
                byte_label(total),
                transform=side.transAxes,
                va="top",
                ha="right",
                fontsize=label_size,
                color=WHITE,
            )
            by_role = {}
            for a in pending:
                by_role[a["role"]] = by_role.get(a["role"], 0) + a["transfer_bytes"]
            side.add_patch(
                Rectangle(
                    (0.065, y - bar_y),
                    0.87,
                    0.033,
                    transform=side.transAxes,
                    facecolor=BG,
                    edgecolor="none",
                )
            )
            x = 0.065
            for role, size in sorted(by_role.items()):
                width = 0.87 * size / scale
                side.add_patch(
                    Rectangle(
                        (x, y - bar_y),
                        width,
                        0.033,
                        transform=side.transAxes,
                        facecolor=ROLE_COLOR.get(role, MUTED),
                        edgecolor="none",
                    )
                )
                x += width
            y -= step
        side.text(
            0.935,
            y + (0.013 if compact else 0.035),
            f"Bar scale: {byte_label(scale)}",
            transform=side.transAxes,
            ha="right",
            fontsize=10,
            color=MUTED,
        )
        return y

    def task_wait_card(self, side, state):
        detail = self.data.task_detail(state["task"])
        instant = state["cursor_ns"]
        y = self.card_text(side, 0.965, detail["label"], size=18)
        y = self.card_text(
            side,
            y,
            f"Boundary {detail['boundary']} · t = {instant / 1e9:.4f} s",
            size=11,
            color=MUTED,
        )
        status = (
            "Upcoming"
            if instant < detail["ready_ns"]
            else "Waiting"
            if instant < detail["start_ns"]
            else "Running"
            if instant < detail["end_ns"]
            else "Complete"
        )
        y = self.card_text(
            side,
            y,
            f"{status} · {duration_label(detail['stall_ns'])} stall",
            size=15,
            color=RED if status == "Waiting" else GREEN,
        )
        y = self.card_text(
            side,
            y,
            f"Task's turn   {detail['ready_ns'] / 1e9:.4f} s\n"
            f"Compute starts   {detail['start_ns'] / 1e9:.4f} s",
            size=11,
        )
        reasons = detail["stall_reasons"]
        blockers = detail["blocking_transfers"]
        if detail["critical_action"] is not None:
            critical = self.data.actions[detail["critical_action"]]
            t = critical["transfer"]
            relation = blockers[0]["relation"]
            if relation == "required_input":
                reason = (
                    f"Waiting for {byte_label(detail['input_fetch_bytes'])} "
                    "of required inputs."
                )
            else:
                purposes = {
                    p
                    for b in blockers
                    if b["relation"] == "storage_reuse"
                    for p in b["purposes"]
                }
                purpose = (
                    "output and workspace addresses"
                    if "task_output" in purposes and "task_workspace" in purposes
                    else "workspace addresses"
                    if "task_workspace" in purposes
                    else "output addresses"
                    if "task_output" in purposes
                    else "reused addresses"
                )
                reason = f"Evictions must release this task's {purpose}."
            y = self.card_text(side, y - 0.01, reason, size=12)
            # The last completion explains when the task can start. Earlier
            # required transfers remain outlined in the lanes as well.
            size_label = byte_label(critical["bytes"])
            if critical["transfer_bytes"] != critical["bytes"]:
                size_label = (
                    f"GPU {size_label} · CPU {byte_label(critical['transfer_bytes'])}"
                )
            y = self.card_text(
                side,
                y,
                f"Last blocker · {size_label}",
                size=11,
                color=MUTED,
            )
            y = self.card_text(
                side,
                y,
                truncate(critical["label"], 65),
                size=12,
                color=ROLE_COLOR.get(critical["role"], MUTED),
            )
            y = self.card_text(
                side,
                y,
                f"{t['direction'].title()} triggered at B{critical['boundary']}\n"
                f"Finishes {t['end_ns'] / 1e9:.4f} s",
                size=11,
            )
            if relation == "storage_reuse":
                r = detail["requirements"]
                y = self.card_text(
                    side,
                    y,
                    f"Outputs {byte_label(r['outputs']['bytes'])}\n"
                    f"Task workspace {byte_label(r['workspace_bytes'])}",
                    size=11,
                    color=MUTED,
                )
            elif detail["later_use_predecessor_bytes"]:
                y = self.card_text(
                    side,
                    y,
                    f"{byte_label(detail['later_use_predecessor_bytes'])} of CPU "
                    "transfers for later tasks went first.",
                    size=11,
                    color=MUTED,
                )
            elif "lane-busy" in t["stall_reasons"]:
                y = self.card_text(
                    side,
                    y,
                    "Earlier transfers occupy this lane until "
                    f"{t['start_ns'] / 1e9:.4f} s.",
                    size=11,
                    color=MUTED,
                )
        elif detail["stall_ns"]:
            labels = {
                "device-capacity": "Waiting for output/workspace allocation capacity.",
                "input-residency": "Required inputs are not yet resident.",
                "memory-reuse": "Waiting for reusable storage.",
            }
            self.card_text(
                side, y, "\n".join(labels.get(r, r) for r in reasons), size=12
            )
        else:
            self.card_text(
                side, y, "This task has no recorded start stall.", size=12, color=MUTED
            )
        self.queue_bars(side, 0.245, instant, compact=True)
