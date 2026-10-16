"""Workload-independent example selection and editable video timing."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from tools.pressurefit_trace.data import short_task


@dataclass
class Timing:
    explanation: float = 12
    motion: float = 10
    build: float = 30
    sweep: float = 40
    hold: float = 6


def action_examples(data):
    chosen = []
    for kind in ("release", "evict"):
        candidates = []
        for action in data.actions:
            if action["kind"] != kind or action["bytes"] < 2**20:
                continue
            detail = data.action_detail(action["index"])
            if detail["next_task"] is None or detail["later_fetch"] is None:
                continue
            gap = detail["next_task"] - action["boundary"]
            # A readable local example: a real gap, a later fetch, and a
            # meaningful size. Avoid an almost-full-step empty focus panel.
            candidates.append(
                (
                    (not 2 <= gap <= 40, gap, -action["bytes"], action["index"]),
                    action["index"],
                )
            )
        if candidates:
            chosen.append(min(candidates)[1])
    fetch = next(
        (
            b["action"]
            for task in execution_examples(data)
            for b in data.task_detail(task)["blocking_transfers"]
            if b["relation"] == "required_input"
        ),
        None,
    )
    if fetch is None:
        fetch = next((a["index"] for a in data.actions if a["kind"] == "fetch"), None)
    if fetch is not None:
        chosen.append(fetch)
    return list(dict.fromkeys(chosen))


def execution_examples(data):
    """Select exposed task stalls, never harmless transfer queue waits."""
    candidates = {"input": [], "storage": [], "prefetch": []}
    for i, task in enumerate(data.tasks):
        if task["start_ns"] <= task["ready_ns"]:
            continue
        detail = data.task_detail(i)
        action_index = detail["critical_action"]
        if action_index is None:
            continue
        action = data.actions[action_index]
        t = action["transfer"]
        if t["end_ns"] != task["start_ns"]:
            continue
        relation = detail["blocking_transfers"][0]["relation"]
        if relation == "storage_reuse" and "lane-busy" in t["stall_reasons"]:
            category = "storage"
        elif relation == "required_input":
            upstream = [
                data.actions[e["action"]]
                for e in detail["upstream_storage_dependencies"]
                if e["successor_action"] == action_index
            ]
            if any(
                e["transfer"]["end_ns"] == t["start_ns"]
                and "lane-busy" in e["transfer"]["stall_reasons"]
                for e in upstream
            ):
                category = "prefetch"
            elif "lane-busy" in t["stall_reasons"]:
                category = "input"
            else:
                continue
        else:
            continue
        candidates[category].append(i)
    chosen = []
    for options in candidates.values():
        if options:
            chosen.append(
                max(
                    options,
                    key=lambda i: (
                        data.actions[data.task_detail(i)["critical_action"]]["bytes"]
                        >= 2**20,
                        data.task_detail(i)["later_use_predecessor_bytes"] == 0,
                        len(data.task_detail(i)["blocking_transfers"]) == 1,
                        data.task_detail(i)["stall_ns"],
                        -i,
                    ),
                )
            )
    return chosen


def time_window(data, start, end, *, pad=0.15):
    duration = max(end - start, min(data.end_ns / 25, 10_000_000))
    return [
        max(0, int(start - duration * pad)),
        min(data.end_ns, int(end + duration * pad)),
    ]


def action_window(data, index, *, full_lifetime=False):
    detail = data.action_detail(index)
    a = data.actions[index]
    start = a["trigger_ns"]
    end = a["transfer"]["end_ns"] if a["transfer"] else start
    if full_lifetime:
        prior, following = detail["prior_task"], detail["next_task"]
        if prior is not None:
            start = min(start, data.tasks[prior]["start_ns"])
        if following is not None:
            end = max(end, data.tasks[following]["end_ns"])
    else:
        following = detail["next_task"]
        if (
            following is not None
            and data.tasks[following]["start_ns"] - end < data.end_ns / 20
        ):
            end = max(end, data.tasks[following]["end_ns"])
    return time_window(data, start, end)


def build_story(
    data, replay, mode, *, timing=None, actions=None, placements=None, tasks=None
):
    timing = timing or Timing()
    scenes = []

    def add(kind, seconds, title, **kw):
        scenes.append(
            {
                "kind": kind,
                "seconds": seconds,
                "title": title,
                "window_ns": [0, data.end_ns],
                "cursor_start_ns": 0,
                "cursor_end_ns": 0,
                **kw,
            }
        )

    if mode == "emission-placement":
        actions = action_examples(data) if actions is None else actions
        placements = replay.examples() if placements is None else placements
        add("emission-overview", timing.hold, "From residency to an executable plan")
        for index in actions:
            detail = data.action_detail(index)
            title = {
                "release": "A current spill copy permits release",
                "evict": "Preserve a modified value before reuse",
                "fetch": "A fetch trigger reserves its destination",
                "write_back": "Refresh the spill copy",
            }[detail["kind"]]
            window = action_window(data, index, full_lifetime=True)
            add(
                "emission",
                timing.explanation,
                title,
                action=index,
                window_ns=window,
                cursor_start_ns=detail["trigger_ns"],
                cursor_end_ns=detail["trigger_ns"],
            )
            add(
                "emission",
                timing.motion,
                "Follow the value to its next use",
                action=index,
                window_ns=window,
                cursor_start_ns=window[0],
                cursor_end_ns=window[1],
            )
        add(
            "placement-intro",
            timing.hold,
            "Give every allocation a physical address",
            placed=0,
        )
        last = 0
        for index in placements:
            rank = int(replay.ranks[index])
            if rank > last:
                add(
                    "placement-build",
                    max(1, timing.build / (len(placements) + 1)),
                    "Place allocations in priority order",
                    placed_start=last,
                    placed_end=rank,
                )
            p = data.leases[index]
            detail = replay.detail(index)
            start = p["predicted_start_ns"]
            previous = detail["reused_predecessor_indices"]
            if previous:
                last_owner = max(
                    previous, key=lambda i: data.leases[i]["predicted_end_ns"]
                )
                start = min(start, data.leases[last_owner]["predicted_start_ns"])
            window = time_window(data, start, p["predicted_end_ns"], pad=0.35)
            add(
                "placement-detail",
                timing.explanation,
                "Find the lowest aligned gap",
                lease=index,
                placed=rank,
                window_ns=window,
                reveal=False,
            )
            add(
                "placement-detail",
                timing.motion,
                "Commit this allocation at its fixed address",
                lease=index,
                placed=rank + 1,
                window_ns=window,
                reveal=True,
            )
            last = rank + 1
        if last < len(data.leases):
            add(
                "placement-build",
                timing.build,
                "Complete the physical layout",
                placed_start=last,
                placed_end=len(data.leases),
            )
        add(
            "placement-sweep",
            timing.sweep,
            "Storage is reused as lifetimes end",
            placed=len(data.leases),
            cursor_start_ns=0,
            cursor_end_ns=data.end_ns,
            pan=True,
        )
        add(
            "placement-final",
            timing.hold,
            "One fixed layout for the entire program",
            placed=len(data.leases),
            cursor_start_ns=data.end_ns,
            cursor_end_ns=data.end_ns,
        )
    elif mode == "execution":
        tasks = execution_examples(data) if tasks is None else tasks
        add(
            "execution-overview",
            timing.hold,
            "Compute and transfers share one timeline",
        )
        for index in tasks:
            detail = data.task_detail(index)
            task = data.tasks[index]
            if detail["stall_ns"] <= 0:
                raise ValueError(
                    "execution examples must name tasks with a recorded stall"
                )
            critical = detail["critical_action"]
            start = task["ready_ns"]
            if critical is not None:
                start = min(start, data.actions[critical]["trigger_ns"])
            for edge in detail["upstream_storage_dependencies"]:
                start = min(start, data.actions[edge["action"]]["trigger_ns"])
            # Include the preceding compute task and the selected task's
            # actual run. Very short tasks retain their exact width.
            if index:
                start = min(start, data.tasks[index - 1]["start_ns"])
            window = time_window(data, start, task["end_ns"])
            title = f"Why {short_task(task)} cannot start"
            mid = (task["ready_ns"] + task["start_ns"]) // 2
            add(
                "execution-detail",
                timing.explanation,
                title,
                task=index,
                window_ns=window,
                cursor_start_ns=mid,
                cursor_end_ns=mid,
            )
            add(
                "execution-detail",
                timing.motion * 2,
                "Follow the blockers until compute can start",
                task=index,
                window_ns=window,
                cursor_start_ns=window[0],
                cursor_end_ns=window[1],
            )
        add(
            "execution-sweep",
            timing.sweep,
            "Follow the whole program",
            pan=True,
            cursor_start_ns=0,
            cursor_end_ns=data.end_ns,
        )
        add(
            "execution-overview",
            timing.hold,
            "The complete simulated step",
            cursor_start_ns=data.end_ns,
            cursor_end_ns=data.end_ns,
        )
    else:
        raise ValueError("unknown video mode")
    elapsed = 0
    for scene in scenes:
        scene["video_start_seconds"] = elapsed
        elapsed += scene["seconds"]
    return {
        "schema": "shadowspill.plan_video.storyboard/v1",
        "mode": mode,
        "timing": asdict(timing),
        "seconds": elapsed,
        "scenes": scenes,
    }


def state_at(scene, fraction, data, *, window_tasks=8):
    fraction = float(np.clip(fraction, 0, 1))
    cursor = round(
        scene["cursor_start_ns"]
        + fraction * (scene["cursor_end_ns"] - scene["cursor_start_ns"])
    )
    state = {**scene, "cursor_ns": cursor, "fraction": fraction}
    if "placed_start" in scene:
        state["placed"] = round(
            scene["placed_start"]
            + fraction**1.5 * (scene["placed_end"] - scene["placed_start"])
        )
    if scene.get("pan"):
        # A fixed time scale makes the sweep a continuous pan. Resizing the
        # window at task transitions produces sudden zooms around tiny tasks.
        # The requested task count uses the full-step average duration,
        # including waits; detailed examples have their own fitted windows.
        width = min(
            data.end_ns,
            max(data.end_ns / 100, data.end_ns * window_tasks / len(data.tasks)),
        )
        start = min(max(0, cursor - width * 0.4), max(0, data.end_ns - width))
        state["window_ns"] = [int(start), int(min(data.end_ns, start + width))]
    return state
