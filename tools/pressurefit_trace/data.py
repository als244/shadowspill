"""Trace analysis shared by the cut-trajectory renderer.

Only the chosen candidate is enriched; a collection may contain many other
large candidates. All event coordinates and scores come from the binary trace.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from shadowspill.diagnostics.occupancy import ROLE_PRIORITY
from shadowspill.diagnostics.pressurefit_trace import PressureFitTrace, boundary_labels

GIB = 2**30
MIB = 2**20
ROLE_COLORS = {
    "parameter": "#5d94ff",
    "gradient": "#35cbb5",
    "optimizer_state": "#9655ff",
    "activation": "#efbc62",
    "input": "#e48c7c",
    "buffer": "#98a9bf",
    "control": "#98a9bf",
    "output": "#efbc62",
    "unattributed": "#98a9bf",
}


def byte_label(value: float) -> str:
    for unit, scale in [("GiB", GIB), ("MiB", MIB), ("KiB", 1024)]:
        if abs(value) >= scale:
            number = value / scale
            return (
                f"{number:,.0f} {unit}"
                if number.is_integer()
                else f"{number:,.2f} {unit}"
            )
    return f"{value:,.0f} B"


def short_task(task: dict[str, Any], *, phase: str | None = None) -> str:
    phase = phase or task.get("phase", "task")
    prefix = {
        "forward": "F",
        "backward": "B",
        "recompute": "R",
        "optimizer": "Opt",
    }.get(phase, phase)
    if task.get("stage_index") is None:
        if phase in ("optimizer", "recompute"):
            return prefix
        return task.get("label", task["task_id"])
    return f"{prefix}{task['stage_index']} · mb{task.get('microbatch', '?')}"


def annotate_recompute(tasks, program):
    """Compare paired backward alternatives without guessing from task names.

    A graph-pair group contains one backward task in each variant. Groups with
    several backward tasks have no unambiguous pairing and remain unshaded.
    Keep the signed difference for analysis; a negative profiling difference
    cannot occupy a positive interval in the drawing.
    """
    selected = {task["task_id"]: task for task in tasks}
    definitions = {task["task_id"]: task for task in program["tasks"]}
    profiles = {p["profile_id"]: p for p in program["profiles"]}
    for group in program.get("task_alternative_groups", ()):
        options = {option["option_id"]: option for option in group["options"]}
        if not {"save", "recompute"} <= options.keys():
            continue
        backward = {
            variant: [
                definitions[task_id]
                for task_id in options[variant]["active_task_ids"]
                if definitions[task_id]["phase"] == "backward"
            ]
            for variant in ("save", "recompute")
        }
        if any(len(value) != 1 for value in backward.values()):
            continue
        save, recompute = backward["save"][0], backward["recompute"][0]
        row = selected.get(recompute["task_id"])
        if row is None:
            continue
        saved_ns = int(profiles[save["profile_id"]]["runtime_ns"])
        recompute_ns = int(profiles[recompute["profile_id"]]["runtime_ns"])
        if recompute_ns != row["profiled_duration_ns"]:
            raise ValueError("selected backward timing differs from its profile")
        row.update(
            variant="recompute",
            recompute_group_id=group["group_id"],
            save_backward_task_id=save["task_id"],
            save_backward_duration_ns=saved_ns,
            recompute_extra_ns=recompute_ns - saved_ns,
            recompute_prefix_ns=max(0, recompute_ns - saved_ns),
        )


def task_memory(task, member_alias, alias_sizes, workspace_bytes):
    """Required storage by declaration, with shared/in-place aliases counted once.

    Counts are distinct declared objects; bytes are their full storage aliases,
    as charged by PressureFit. Categories may overlap. The total unions their
    objects and aliases before adding the task's profiled workspace.
    """
    groups = {
        "inputs": set(task["inputs"]),
        "outputs": set(task["outputs"]),
        "mutated": {m["object_id"] for m in task["mutations"]},
    }

    def footprint(objects):
        aliases = {member_alias[object_id] for object_id in objects}
        return {
            "object_count": len(objects),
            "alias_count": len(aliases),
            "bytes": sum(int(alias_sizes[a]) for a in aliases),
        }

    result = {name: footprint(objects) for name, objects in groups.items()}
    total = footprint(set().union(*groups.values()))
    result["overlap_bytes"] = (
        sum(item["bytes"] for item in result.values()) - total["bytes"]
    )
    total["bytes"] += int(workspace_bytes)
    result.update(
        workspace_bytes=int(workspace_bytes),
        total=total,
        byte_basis="unique alias storage",
    )
    return result


@dataclass
class Trajectory:
    directory: Path
    candidate_index: int
    candidate: dict[str, Any]
    trace: PressureFitTrace
    aliases: list[dict[str, Any]]
    tasks: list[dict[str, Any]]
    edges: np.ndarray
    uses: list[list[int]]
    input_uses: list[set[int]]
    productions: list[set[int]]
    minimum_bytes: int
    order: np.ndarray
    y_edges: np.ndarray
    y_rank: np.ndarray
    axis: str = "compute"
    pressure_accounting: dict[str, Any] | None = None
    collection_context: dict[str, Any] | None = None

    @property
    def compute_seconds(self):
        return sum(t["profiled_duration_ns"] for t in self.tasks) / 1e9

    def select_axis(self, axis):
        """Change drawing coordinates, preserving filtering and planner state."""
        if axis not in ("compute", "task"):
            raise ValueError("axis must be compute or task")
        self.axis = axis
        self.edges = (
            np.arange(len(self.tasks) + 1, dtype=float)
            if axis == "task"
            else np.concatenate(
                (
                    [0.0],
                    np.cumsum([t["profiled_duration_ns"] / 1e9 for t in self.tasks]),
                )
            )
        )

    @classmethod
    def load(cls, directory: Path, candidate: str = "winner") -> Trajectory:
        directory = directory.resolve()
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest["status"] != "complete":
            raise ValueError("only complete trace recordings can be rendered")
        candidates = manifest["candidates"]
        if candidate == "winner":
            answer = manifest["answer"]
            matches = [
                i
                for i, c in enumerate(candidates)
                if (c["selection_id"], c["candidate_id"])
                == (answer["selection_id"], answer["candidate_id"])
            ]
            if not matches:
                raise ValueError(
                    "this recording does not contain its overall winner; "
                    "select an index"
                )
            index = matches[0]
        else:
            index = int(candidate)
        if not 0 <= index < len(candidates):
            raise ValueError("candidate index is out of range")
        chosen = candidates[index]
        context = (
            manifest["contexts"][chosen["context_index"]]
            if "context_index" in chosen
            else chosen
        )
        aliases = context["aliases"]
        trace = PressureFitTrace.read(directory / chosen["file"])
        if len(aliases) != len(trace.alias_sizes):
            raise ValueError("alias context does not match trace")
        sidecar = directory / "semantic_labels.json"
        semantics = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        if semantics and semantics["program_digest"] != manifest["program_digest"]:
            raise ValueError("semantic annotations belong to a different program")
        boundaries = boundary_labels(chosen["tasks"], semantics.get("tasks", {}))
        tasks = [b["before"] for b in boundaries[:-1]]
        task_lookup = {t["task_id"]: i for i, t in enumerate(tasks)}
        program = json.loads((directory / manifest["program"]).read_text())
        program_tasks = {t["task_id"]: t for t in program["tasks"]}
        for alias in aliases:
            for member in alias["objects"]:
                meta = semantics.get("objects", {}).get(member["object_id"], {})
                label = meta.get("label")
                if not label:
                    origins = [
                        o
                        for o in meta.get("origins", ())
                        if o["task_id"] in task_lookup
                    ]
                    producer = member["producer"]
                    if not origins and producer in task_lookup:
                        outputs = program_tasks[producer]["outputs"]
                        if member["object_id"] in outputs:
                            origins = [
                                {
                                    "task_id": producer,
                                    "output_index": outputs.index(member["object_id"]),
                                }
                            ]
                    if origins:
                        origin = origins[0]
                        label = (
                            f"{member['role']} output {origin['output_index']} · "
                            f"{short_task(tasks[task_lookup[origin['task_id']]])}"
                        )
                member["label"] = (
                    label or f"{member['category']} · {member['object_id']}"
                )
            if alias["objects"]:
                # Persistent state may also appear as saved activation views.
                # Normalize older recordings too: the alias names its storage,
                # while member labels still identify each task's own view.
                alias["objects"].sort(
                    key=lambda member: (
                        ROLE_PRIORITY.get(member["role"], len(ROLE_PRIORITY)),
                        -member["size_bytes"],
                        member["object_id"],
                    )
                )
                primary = alias["objects"][0]
                alias.update(
                    label=primary["label"],
                    role=primary["role"],
                    category=primary["category"],
                )
            alias["shared_members"] = len(alias["objects"])
            alias["member_roles"] = sorted({o["role"] for o in alias["objects"]})
        edges = np.concatenate(
            ([0.0], np.cumsum([t["profiled_duration_ns"] / 1e9 for t in tasks]))
        )
        if edges[-1] <= 0:
            raise ValueError(
                "profiled task durations must include positive compute time"
            )

        # Access bands come from the selected task declarations, not from
        # first/last use hulls or a guess based on alias IDs.
        annotate_recompute(tasks, program)
        profiles = {p["profile_id"]: p for p in program["profiles"]}
        member_alias = {
            o["object_id"]: a for a, row in enumerate(aliases) for o in row["objects"]
        }
        inputs = [set() for _ in aliases]
        members = {o["object_id"]: o for a in aliases for o in a["objects"]}
        for member in members.values():
            member["reference_positions"] = []
        productions = [set() for _ in aliases]
        for i, row in enumerate(tasks):
            task = program_tasks[row["task_id"]]
            row["workspace_bytes"] = profiles[task["profile_id"]]["workspace_bytes"]
            row["required_memory"] = task_memory(
                task, member_alias, trace.alias_sizes, row["workspace_bytes"]
            )
            for object_id in (
                *task["inputs"],
                *(m["object_id"] for m in task["mutations"]),
            ):
                inputs[member_alias[object_id]].add(i)
                members[object_id]["reference_positions"].append(i)
            for object_id in task["outputs"]:
                productions[member_alias[object_id]].add(i)
                members[object_id]["reference_positions"].append(i)
        uses = [sorted(i | p) for i, p in zip(inputs, productions, strict=True)]
        order = np.asarray(
            sorted(
                range(len(aliases)),
                key=lambda a: (uses[a][0] if uses[a] else len(tasks), a),
            )
        )
        y_edges = np.concatenate(
            ([0.0], np.cumsum(trace.alias_sizes[order].astype(float) / GIB))
        )
        rank = np.empty_like(order)
        rank[order] = np.arange(len(order))
        minimum = manifest["search_options"]["generic"][
            "minimum_object_bytes_evict_eligible"
        ]
        result = cls(
            directory,
            index,
            chosen,
            trace,
            aliases,
            tasks,
            edges,
            uses,
            inputs,
            productions,
            minimum,
            order,
            y_edges,
            rank,
        )
        result.filter_rows()
        result.pressure_accounting = context.get("pressure_accounting")
        collection = directory / "collection_context.json"
        if collection.exists():
            result.collection_context = json.loads(collection.read_text())
            if (
                result.collection_context["program_digest"]
                != manifest["program_digest"]
            ):
                raise ValueError("collection context belongs to a different program")
        return result

    def focus_label(self, alias, position):
        """Use local activation views while retaining persistent state names."""
        row = self.aliases[alias]
        if row["role"] != "activation":
            return row["label"]
        members = [m for m in row["objects"] if m["reference_positions"]]
        if not members:
            return row["label"]
        member = min(
            members,
            key=lambda m: min(abs(i - position) for i in m["reference_positions"]),
        )
        return member["label"]

    def reduction_context(self, reduction):
        capacity = self.trace.object_capacity_bytes(reduction)
        reason = self.trace.reduction_reason(reduction)
        if reason is None:
            return {
                "label": "trigger not recorded",
                "recorded": False,
                "object_capacity_bytes": capacity,
            }
        reason = dict(reason, recorded=True, object_capacity_bytes=capacity)
        task, alias = reason["task_position"], reason["alias_index"]
        if task is not None:
            reason["task_label"] = short_task(self.tasks[task])
        if alias is not None:
            reason["alias_label"] = self.aliases[alias]["label"]
        if reason["kind"] == "initial_seed":
            label = "initial seed · execution pool"
        elif reason["kind"] == "capacity_refinement":
            label = (
                "object-pool layout · "
                f"{byte_label(reason['shortfall_bytes'])} shortfall"
            )
        else:
            kind = {
                "simulation_pressure": "pressure repair",
                "admission_pressure": "admission repair",
                "escalation_rollback": "reduced repair request",
            }[reason["kind"]]
            label = f"{reason['pool']} pool · {kind}"
            if task is not None:
                label += f" · {reason['task_label']}"
            if alias is not None:
                # A task-start capacity failure reports its first output;
                # its request covers all new outputs and workspace together.
                prefix = "first output: " if reason["status"] == 23 else ""
                label += f" · {prefix}{reason['alias_label']}"
            elif reason["status"] == 23:
                label += " · outputs + workspace"
        reason["label"] = label
        return reason

    def next_use(self, row):
        """First task referencing the chosen alias after its removed gap."""
        alias = int(row[4])
        resume = int(row[6:7].view("<i8")[0]) + 2
        tasks = [t for t in self.input_uses[alias] if t >= resume]
        tasks.extend(t for t in self.productions[alias] if t + 1 >= resume)
        return min(tasks) if tasks else None

    def departure_position(self, row):
        """Boundary where the removed display interval begins.

        An input anchor retains the alias through that task; an output anchor
        is already after its producing task. Initial cuts begin at boundary 0.
        This is a residency boundary, not a scheduled transfer timestamp.
        """
        alias = int(row[4])
        retained_end = int(row[5:6].view("<i8")[0])
        return max(0, retained_end + int(retained_end in self.input_uses[alias]))

    def gap_duration_ns(self, row, next_use):
        """Profiled compute between release/eviction and the next task use."""
        return sum(
            t["profiled_duration_ns"]
            for t in self.tasks[self.departure_position(row) : next_use]
        )

    def time_to_use_ns(self, row, next_use):
        """Profiled compute from the trigger boundary to the next task use."""
        return sum(
            t["profiled_duration_ns"] for t in self.tasks[int(row[0]) : next_use]
        )

    def filter_rows(self) -> None:
        """Filter drawing rows only; the trace and pressure arrays stay intact.

        A seed may have no positive-duration idle interval yet still admit a
        boundary split that reduces fetch headroom. Keep every recorded cut
        alias so that such decisions remain visible. Optimizer-state rows also
        remain visible when their only interval is their optimizer task.
        """
        seed = [[] for _ in self.aliases]
        for alias, first, last in self.trace.seed:
            seed[int(alias)].append((int(first), int(last)))
        cut_counts = Counter(
            int(a) for r in self.trace.reductions for a in r.cuts[:, 4]
        )
        for a, row in enumerate(self.aliases):
            # Task intervals do not overlap on the fixed compute-time axis.
            required_seconds = sum(
                self.edges[i + 1] - self.edges[i] for i in self.uses[a]
            )
            resident_seconds = sum(
                end - start for start, end in self.intervals(a, seed[a])
            )
            has_idle = resident_seconds > required_seconds + 1e-12
            size = int(self.trace.alias_sizes[a])
            reason = (
                "below eviction threshold"
                if size < self.minimum_bytes
                else "required use only"
                if not has_idle
                and not cut_counts[a]
                and row["role"] != "optimizer_state"
                else None
            )
            row.update(
                alias_index=a,
                size_bytes=size,
                cut_count=cut_counts[a],
                first_reference=self.uses[a][0] if self.uses[a] else None,
                idle_seconds=max(0, resident_seconds - required_seconds),
                hidden_reason=reason,
            )
        self.order = np.asarray(
            sorted(
                (
                    a
                    for a, row in enumerate(self.aliases)
                    if row["hidden_reason"] is None
                ),
                key=lambda a: (self.uses[a][0] if self.uses[a] else len(self.tasks), a),
            ),
            dtype=np.intp,
        )
        self.y_edges = np.concatenate(
            ([0.0], np.cumsum(self.trace.alias_sizes[self.order].astype(float) / GIB))
        )
        self.y_rank = np.full(len(self.aliases), -1, dtype=np.intp)
        self.y_rank[self.order] = np.arange(len(self.order))
        hidden_cuts = [a for a in cut_counts if self.y_rank[a] < 0]
        if hidden_cuts:
            raise ValueError(
                "trace cut aliases below its recorded eviction threshold: "
                f"{hidden_cuts[:5]}"
            )

    def focus_aliases(
        self, position: int, selected: int | None, context: int, limit: int
    ):
        """Prefer main roles, required rows by size, then nearby references.

        The selected alias is pinned even when its next use is outside the
        window or has an "other" role. Other-role neighbors only fill unused
        rows. Return the full local list as well as its legible viewport.
        """
        lo, hi = (
            max(0, position - context),
            min(len(self.tasks), position + context + 1),
        )
        relevant = {}
        role_priority = {
            "parameter": 0,
            "activation": 1,
            "gradient": 2,
            "optimizer_state": 3,
        }
        for alias in self.order:
            local = [i for i in self.uses[alias] if lo <= i < hi]
            if local:
                relevant[int(alias)] = min(local, key=lambda i: (abs(i - position), i))
        ranked = sorted(
            relevant,
            key=lambda a: (
                self.aliases[a]["role"] not in role_priority,
                position not in self.uses[a],
                -int(self.trace.alias_sizes[a]) if position in self.uses[a] else 0,
                role_priority.get(self.aliases[a]["role"], 4)
                if position in self.uses[a]
                else 0,
                abs(relevant[a] - position),
                relevant[a],
                self.uses[a][0],
                a,
            ),
        )
        if selected is not None and self.y_rank[selected] >= 0:
            ranked = [selected, *(a for a in ranked if a != selected)]
        return ranked[:limit], ranked

    def row_filter_summary(self):
        return {
            "minimum_bytes": self.minimum_bytes,
            "visible_aliases": len(self.order),
            "required_use_visible_roles": ["optimizer_state"],
            "hidden": dict(
                Counter(a["hidden_reason"] for a in self.aliases if a["hidden_reason"])
            ),
            "order": "first task reference (input, mutation, or output); "
            "alias index breaks ties",
            "pressure_includes_all_aliases": True,
        }

    def intervals(self, alias: int, spans) -> list[tuple[float, float]]:
        """Project logical boundary spans through the tasks using those anchors.

        Input anchors are before the task: extend the display through that
        task's end. Fresh outputs reserve memory during the producing task:
        extend their display back to that task's start. No transfer timestamps
        are inferred. Touching spans remain separate to preserve split cuts.
        """
        intervals = []
        for first, last in spans:
            start = first - 1 if first - 1 in self.productions[alias] else first
            end = last + 1 if last in self.input_uses[alias] else last
            intervals.append((float(self.edges[start]), float(self.edges[end])))
        return intervals

    def assert_required_covered(self, spans) -> None:
        for alias, tasks in enumerate(self.uses):
            intervals = self.intervals(alias, spans[alias])
            for task in tasks:
                start, end = self.edges[task : task + 2]
                if not any(
                    a <= start + 1e-12 and b >= end - 1e-12 for a, b in intervals
                ):
                    raise ValueError(
                        f"required task interval lost: alias {alias}, task {task}"
                    )

    def distribution(self) -> dict[str, Any]:
        sizes = self.trace.alias_sizes.astype(float)
        bins = [0, 1024, MIB, 10 * MIB, 100 * MIB, GIB, float("inf")]
        names = [
            "< 1 KiB",
            "1 KiB-1 MiB",
            "1-10 MiB",
            "10-100 MiB",
            "100 MiB-1 GiB",
            "≥ 1 GiB",
        ]
        buckets = []
        for low, high, name in zip(bins[:-1], bins[1:], names, strict=True):
            chosen = (sizes >= low) & (sizes < high)
            buckets.append(
                {
                    "range": name,
                    "count": int(chosen.sum()),
                    "bytes": int(sizes[chosen].sum()),
                    "byte_share": float(sizes[chosen].sum() / sizes.sum()),
                }
            )
        return {
            "alias_count": len(sizes),
            "sum_alias_bytes": int(sizes.sum()),
            "note": "Summed alias sizes are not simultaneous live memory.",
            "quantiles_bytes": dict(
                zip(
                    ["min", "p10", "p25", "median", "p75", "p90", "p95", "p99", "max"],
                    np.quantile(
                        sizes, [0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]
                    ).tolist(),
                    strict=True,
                )
            ),
            "buckets": buckets,
            "shared_aliases": sum(len(a["objects"]) > 1 for a in self.aliases),
            "mixed_role_aliases": sum(len(a["member_roles"]) > 1 for a in self.aliases),
        }


def analyze(data: Trajectory, output: Path) -> dict[str, Any]:
    from .explain import decision
    from .pressure import ROLE_GROUPS, PressureComposition

    output.mkdir(parents=True, exist_ok=True)
    data.trace.verify()
    composition = PressureComposition(data)
    role_arrays = {
        "roles": np.asarray([name for name, _, _ in ROLE_GROUPS]),
        "resident_reservations": composition.resident_roles,
        "workspace": composition.workspace,
        "budget_excluding_resident_bytes": np.asarray(composition.flat_budget),
    }
    report = {
        "source": str(data.directory),
        "candidate_index": data.candidate_index,
        "candidate_id": data.candidate["candidate_id"],
        "collection_context": data.collection_context,
        "clock": "profiled compute time",
        "compute_seconds": data.compute_seconds,
        "recompute_timing": {
            "matched_backward_tasks": sum(
                "recompute_prefix_ns" in task for task in data.tasks
            ),
            "negative_profile_differences": sum(
                task.get("recompute_extra_ns", 0) < 0 for task in data.tasks
            ),
            "prefix_duration_ns": sum(
                task.get("recompute_prefix_ns", 0) for task in data.tasks
            ),
        },
        "distribution": data.distribution(),
        "row_filter": data.row_filter_summary(),
        "answer_step": data.trace.answer_step,
        "answer_reduction": data.trace.answer_reduction,
        "reductions": [],
        "verified": True,
        "pressure_attributed": composition.attributed,
        "resident_slice_bytes": composition.resident_slice,
        "planning_budget_bytes": composition.flat_budget + composition.resident_slice,
    }
    with (
        (output / "cuts.csv").open("w", newline="") as stream,
        (output / "decisions.jsonl").open("w") as decisions,
    ):
        writer = csv.writer(stream)
        writer.writerow(
            [
                "reduction",
                "cut",
                "alias_index",
                "semantic_name",
                "bytes",
                "start_boundary",
                "end_boundary",
                "trigger_position",
                "trigger_task_before",
                "used_before_bytes",
                "used_after_bytes",
                "capacity_bytes",
                "peak_after_bytes",
                "max_excess_after_bytes",
                "exposed_ns",
                "writeback_score",
                "first_difference_vs_runner_up",
            ]
        )
        for rid, reduction in enumerate(data.trace.reductions):
            # The private streaming walk avoids copying every alias at every
            # cut. Snapshots for displayed frames copy only their focus rows.
            walk = data.trace._walk(rid)
            _, spans, pressure, capacity = next(walk)
            data.assert_required_covered(spans)
            start_peak = int(pressure.max())
            role_arrays[f"reduction_{rid:03d}_seed"] = composition.at(
                rid, 0, pressure
            ).copy()
            counts = Counter()
            rows = []
            for index, row in enumerate(reduction.cuts):
                decisions.write(json.dumps(decision(data, rid, index)) + "\n")
                trigger, alias = int(row[0]), int(row[4])
                before = int(pressure[trigger])
                _, spans, pressure, capacity = next(walk)
                composition.display(rid, index + 1, pressure, capacity)
                start, end = (int(x) for x in row[5:7].view("<i8"))
                alternatives = data.trace.comparisons(rid, index)
                max_excess = max(
                    0,
                    int(np.max(pressure.astype(np.int64) - capacity.astype(np.int64))),
                )
                writer.writerow(
                    [
                        rid,
                        index + 1,
                        alias,
                        data.aliases[alias]["label"],
                        int(data.trace.alias_sizes[alias]),
                        start,
                        end,
                        trigger,
                        data.tasks[trigger]["label"]
                        if trigger < len(data.tasks)
                        else "end",
                        before,
                        int(pressure[trigger]),
                        int(capacity[trigger]),
                        int(pressure.max()),
                        max_excess,
                        int(row[7]),
                        int(row[8]),
                        alternatives[0]["criterion"]
                        if alternatives
                        else "only legal cut",
                    ]
                )
                counts[data.aliases[alias]["role"]] += 1
                rows.append([index + 1, int(pressure.max()), max_excess])
            data.assert_required_covered(spans)
            role_arrays[f"reduction_{rid:03d}_final"] = composition.values.copy()
            report["reductions"].append(
                {
                    "reduction": rid,
                    "reason": data.reduction_context(rid),
                    "cuts": len(reduction.cuts),
                    "status": reduction.status,
                    "seed_peak_bytes": start_peak,
                    "final_peak_bytes": int(pressure.max()),
                    "cuts_by_primary_role": dict(counts),
                    "final_max_excess_bytes": max(
                        0,
                        int(
                            np.max(
                                pressure.astype(np.int64) - capacity.astype(np.int64)
                            )
                        ),
                    ),
                }
            )
            np.save(
                output / f"reduction_{rid:03d}_metrics.npy",
                np.asarray(rows, dtype=np.int64),
            )
    np.savez_compressed(output / "pressure_roles.npz", **role_arrays)
    (output / "analysis.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "aliases.json").write_text(json.dumps(data.aliases, indent=2) + "\n")
    (output / "tasks.json").write_text(json.dumps(data.tasks, indent=2) + "\n")
    return report
