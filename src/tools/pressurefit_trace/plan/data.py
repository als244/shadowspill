"""One validated plan, with the same attribution used by the HTML timelines.

Video time is separate from simulated time. No timestamp is inferred from a
PressureFit score. Physical allocations keep the certificate's exact offsets.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from shadowspill.diagnostics.occupancy import (
    ROLE_PRIORITY,
    WORKSPACE,
    Occupancy,
    attribute,
    placement_rows,
)
from shadowspill.planner.annotated_plan import AnnotatedProgramPlan
from tools.pressurefit_trace.data import annotate_recompute, short_task

from .waits import TaskWaits


@dataclass
class EventSeries:
    """Exact half-open interval sums, including simultaneous starts and ends."""

    times: np.ndarray
    values: np.ndarray
    names: tuple[str, ...]

    @classmethod
    def build(cls, intervals, end_ns):
        events = defaultdict(lambda: defaultdict(int))
        names = set()
        for start, end, amount, name in intervals:
            if end <= start:
                continue
            names.add(name)
            events[int(start)][name] += int(amount)
            events[int(end)][name] -= int(amount)
        names = tuple(sorted(names))
        events[0]
        events[int(end_ns)]
        times = np.array(sorted(events), dtype=np.int64)
        values = np.zeros((len(times), len(names)), dtype=np.int64)
        index = {name: i for i, name in enumerate(names)}
        current = np.zeros(len(names), dtype=np.int64)
        for row, instant in enumerate(times):
            for name, delta in events[int(instant)].items():
                current[index[name]] += delta
            values[row] = current
        if np.any(values < 0):
            raise ValueError("negative interval accounting")
        return cls(times, values, names)

    @classmethod
    def occupancy(cls, pool: Occupancy, end_ns):
        return cls.build(
            ((i.start_ns, i.end_ns, i.bytes, i.role) for i in pool.intervals),
            end_ns,
        )

    def at(self, time_ns):
        i = max(0, int(np.searchsorted(self.times, time_ns, side="right")) - 1)
        return dict(zip(self.names, map(int, self.values[i]), strict=True))

    @property
    def peak(self):
        return int(self.values.sum(axis=1).max(initial=0))


class PlanData:
    def __init__(self, raw, *, semantics=None, metadata=None, source=None):
        # The product decoder verifies the program, schedule, certificate,
        # and selected makespan agree. Do this before drawing any evidence.
        restored = AnnotatedProgramPlan.from_dict(raw)
        self.source = str(source) if source else None
        self.program = raw["source_program"]["program"]["value"]
        self.program_digest = restored.program.program.digest
        self.schedule_digest = restored.result.schedule.digest
        self.layout_digest = restored.fixed_layout.digest
        self.schedule = raw["selection"]["schedule"]
        self.layout = raw["physical_admission"]["fixed_layout"]
        self.simulation = raw["simulation"]["result"]
        self.selection = raw["selection"]
        self.end_ns = int(self.simulation["makespan_ns"])
        if self.end_ns <= 0:
            raise ValueError("video requires a positive simulated makespan")
        if len(self.program["devices"]) != 1:
            raise ValueError("plan videos currently support one device")
        self.semantics = semantics or {}
        if (
            self.semantics.get("program_digest", self.program_digest)
            != self.program_digest
        ):
            raise ValueError("semantic labels belong to another program")
        self.metadata = metadata or {}
        if (
            self.metadata.get("schedule_digest", self.schedule_digest)
            != self.schedule_digest
        ):
            raise ValueError("metadata belongs to another schedule")
        self.pool_bytes = int(self.layout["pool_capacity_bytes"])
        self.spill_bytes = int(raw["memory_budgets"]["spill_bytes"])
        self.resident_aliases = set(self.selection["resident_slice"]["aliases"])
        self.policy = self.selection["diagnostics"]["selection"]["candidate_id"]
        stored = {
            **self.selection,
            "simulation_result": self.simulation,
            "simulation": {"spill_capacity_bytes": self.spill_bytes},
            "admission_certificate": {"layout": self.layout},
        }
        self.occupancy = attribute(stored, self.program)
        self.facts = self.occupancy.facts
        self.alias_members = defaultdict(list)
        for member in self.facts.objects.values():
            self.alias_members[member.alias_group_id].append(member)
        self.task_index = dict(self.facts.task_order)
        definitions = {t["task_id"]: t for t in self.program["tasks"]}
        profiles = {p["profile_id"]: p for p in self.program["profiles"]}
        intervals = {t["task_id"]: t for t in self.simulation["task_intervals"]}
        self.tasks = []
        for task_id in self.task_index:
            definition = definitions[task_id]
            self.tasks.append(
                {
                    **definition,
                    **intervals[task_id],
                    **self.semantics.get("tasks", {}).get(task_id, {}),
                    "profiled_duration_ns": int(
                        profiles[definition["profile_id"]]["runtime_ns"]
                    ),
                }
            )
        annotate_recompute(self.tasks, self.program)
        self.task_by_id = {t["task_id"]: t for t in self.tasks}
        self.object_alias = {
            o["object_id"]: o["alias_group_id"] for o in self.program["objects"]
        }
        self.uses = defaultdict(list)
        self.writes = defaultdict(list)
        self.reads = defaultdict(list)
        for i, task in enumerate(self.tasks):
            reads = {self.object_alias[o] for o in task["inputs"]}
            writes = {self.object_alias[o] for o in task["outputs"]}
            mutated = {self.object_alias[m["object_id"]] for m in task["mutations"]}
            for alias in reads | writes | mutated:
                self.uses[alias].append(i)
            for alias in reads | mutated:
                self.reads[alias].append(i)
            for alias in writes | mutated:
                self.writes[alias].append(i)
        self.leases = placement_rows(self.layout)
        self.lease_index = {p["lease_id"]: i for i, p in enumerate(self.leases)}
        for lease in self.leases:
            alias = lease["alias_group_id"]
            if alias is None or lease["purpose"] == "task_workspace":
                lease["role"] = WORKSPACE
                task = self.task_by_id.get(lease["task_id"])
                lease["label"] = "Task workspace" + (
                    f" · {short_task(task)}" if task else ""
                )
            else:
                lease["label"], lease["role"] = self.alias_label(
                    alias, lease["predicted_start_ns"]
                )
        self.actions = []
        transfers = defaultdict(deque)
        for transfer in sorted(
            self.simulation["transfer_intervals"], key=lambda t: t["sequence"]
        ):
            transfers[
                (
                    transfer["alias_group_id"],
                    transfer["kind"],
                    transfer["trigger_task_id"],
                )
            ].append(transfer)
        for i, action in enumerate(self.schedule["actions"]):
            trigger = action["trigger_task_id"]
            key = (action["alias_group_id"], action["kind"], trigger)
            transfer = transfers[key].popleft() if transfers[key] else None
            instant = int(self.task_by_id[trigger]["end_ns"])
            label, role = self.alias_label(action["alias_group_id"], instant)
            self.actions.append(
                {
                    **action,
                    "index": i,
                    "boundary": self.task_index[trigger] + 1,
                    "trigger_ns": instant,
                    "transfer": transfer,
                    "label": label,
                    "role": role,
                    "bytes": self.facts.alias_size[action["alias_group_id"]],
                }
            )
        if any(transfers.values()):
            raise ValueError("a simulated transfer has no scheduled action")
        self.transfers = [a for a in self.actions if a["transfer"] is not None]
        self.alias_actions = defaultdict(list)
        for action in self.actions:
            self.alias_actions[action["alias_group_id"]].append(action)
        for action in self.transfers:
            t = action["transfer"]
            if not action["trigger_ns"] <= t["start_ns"] <= t["end_ns"]:
                raise ValueError("transfer timestamps precede their trigger")
        self.execution = EventSeries.occupancy(self.occupancy.execution, self.end_ns)
        self.spill = EventSeries.occupancy(self.occupancy.spill, self.end_ns)
        self.pending_count = EventSeries.build(
            (
                (
                    a["trigger_ns"],
                    a["transfer"]["start_ns"],
                    1,
                    a["transfer"]["direction"],
                )
                for a in self.transfers
            ),
            self.end_ns,
        )
        self.pending_bytes = EventSeries.build(
            (
                (
                    a["trigger_ns"],
                    a["transfer"]["start_ns"],
                    a["bytes"],
                    a["transfer"]["direction"],
                )
                for a in self.transfers
            ),
            self.end_ns,
        )
        self.waits = TaskWaits(self, raw["simulation"].get("admission") or {})

    def task_detail(self, index):
        return self.waits.detail(index)

    @classmethod
    def load(cls, path: Path, *, semantics=None, metadata=None):
        path = path.resolve()

        def read(optional):
            return json.loads(Path(optional).read_text()) if optional else None

        return cls(
            json.loads(path.read_text()),
            semantics=read(semantics),
            metadata=read(metadata),
            source=path,
        )

    def object_label(self, object_id):
        meta = self.semantics.get("objects", {}).get(object_id, {})
        if meta.get("label"):
            return meta["label"]
        obj = self.facts.objects[object_id]
        origins = [
            o for o in meta.get("origins", ()) if o["task_id"] in self.task_index
        ]
        if not origins and obj.producer in self.task_index:
            task = self.task_by_id[obj.producer]
            if object_id in task["outputs"]:
                origins = [
                    {
                        "task_id": obj.producer,
                        "output_index": task["outputs"].index(object_id),
                    }
                ]
        if origins:
            origin = origins[0]
            return (
                f"{obj.role} output {origin['output_index']} · "
                f"{short_task(self.task_by_id[origin['task_id']])}"
            )
        return f"{obj.role} · {object_id}"

    def alias_label(self, alias, time_ns):
        # Persistent state names its storage even when a saved activation is
        # another view of it. For reused step storage, use its current owner.
        candidates = self.alias_members[alias]
        holder = self.facts.occupant(alias, self.occupancy.clock.task_start_ns, time_ns)
        persistent = [o for o in candidates if o.persistence == "checkpoint"]
        if persistent:
            holder = min(
                persistent, key=lambda o: (ROLE_PRIORITY.get(o.role, 99), -o.size_bytes)
            )
        if holder is None:
            return alias, "unattributed"
        return self.object_label(holder.object_id), holder.role

    def queue(self, direction, time_ns):
        return sorted(
            (
                a
                for a in self.transfers
                if a["transfer"]["direction"] == direction
                and a["trigger_ns"] <= time_ns < a["transfer"]["start_ns"]
            ),
            key=lambda a: a["transfer"]["sequence"],
        )

    def action_detail(self, index):
        a = self.actions[index]
        alias, trigger = a["alias_group_id"], a["boundary"] - 1
        prior = [i for i in self.uses[alias] if i <= trigger]
        following = [i for i in self.reads[alias] if i > trigger]
        next_use = following[0] if following else None
        earlier = [x for x in self.alias_actions[alias] if x["index"] < index]
        departure = next(
            (x for x in reversed(earlier) if x["kind"] in ("release", "evict")), None
        )
        refreshed = max(
            (
                x["boundary"] - 1
                for x in earlier
                if x["kind"] in ("fetch", "evict", "write_back")
            ),
            default=-1,
        )
        dirty = any(refreshed < i <= trigger for i in self.writes[alias])
        later_fetch = next(
            (
                x["index"]
                for x in self.alias_actions[alias]
                if x["index"] > index and x["kind"] == "fetch"
            ),
            None,
        )
        # Positions label boundaries before task p; actions run after their
        # trigger task, so trigger index i is displayed as Boundary i+1.
        window = None
        if a["kind"] == "fetch":
            lo = 1 if departure is None else departure["boundary"] + 1
            hi = next_use if next_use is not None else len(self.tasks)
            if alias in self.resident_aliases:
                lo = hi = a["boundary"]
            window = [lo, hi]
            if not lo <= a["boundary"] <= hi:
                raise ValueError(
                    f"action {index} lies outside its reconstructed fetch window"
                )
        return {
            **a,
            "prior_task": prior[-1] if prior else None,
            "next_task": next_use,
            "later_fetch": later_fetch,
            "dirty_since_refresh": dirty,
            "legal_fetch_boundaries": window,
        }

    def summary(self):
        return {
            "schema": "shadowspill.plan_video/v1",
            "source": self.source,
            "program_digest": self.program_digest,
            "schedule_digest": self.schedule_digest,
            "layout_digest": self.layout_digest,
            "clock": "simulated",
            "makespan_ns": self.end_ns,
            "tasks": len(self.tasks),
            "actions": dict(Counter(a["kind"] for a in self.actions)),
            "allocations": len(self.leases),
            "pool_capacity_bytes": self.pool_bytes,
            "fixed_slice_bytes": self.layout["fixed_slice_bytes"],
            "required_bytes": self.layout["required_bytes"],
            "execution_peak_bytes": self.execution.peak,
            "spill_peak_bytes": self.spill.peak,
            "pending_peak_count": self.pending_count.peak,
            "pending_peak_bytes": self.pending_bytes.peak,
            "pending_definition": (
                "triggered but not started; excludes copying transfers; "
                "includes admission/dependency waits"
            ),
            "metadata": self.metadata,
        }
