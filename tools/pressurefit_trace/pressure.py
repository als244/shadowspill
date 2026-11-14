"""Partition recorded reducer pressure by alias role, retaining every byte.

The seed charge follows residency/spans.c: fetch headroom, final-boundary
release, and output reservations. Subsequent updates come directly from the
trace's pressure operations, attributed to the alias of the selected cut.
"""

from __future__ import annotations

import numpy as np

from .data import ROLE_COLORS

ROLE_GROUPS = (
    ("parameter", "Parameters", ROLE_COLORS["parameter"]),
    ("gradient", "Gradients", ROLE_COLORS["gradient"]),
    ("optimizer_state", "Optimizer state", ROLE_COLORS["optimizer_state"]),
    ("activation", "Activations", ROLE_COLORS["activation"]),
    ("input", "Inputs", ROLE_COLORS["input"]),
    ("other", "Other roles", "#98a9bf"),
    ("workspace", "Task workspace", "#e77bb7"),
    ("repair", "Repair / reserve", "#c6cbd5"),
)


class PressureComposition:
    """Incremental role totals, checked against the independent pressure replay."""

    def __init__(self, data, *, include_resident=False):
        self.data = data
        self.attributed = data.pressure_accounting is not None
        self.include_resident = include_resident
        if include_resident and not self.attributed:
            raise ValueError(
                "total pressure needs pressure_accounting metadata; "
                "record the candidate again or use --pressure-view reducer"
            )
        roles = {name: i for i, (name, _, _) in enumerate(ROLE_GROUPS)}
        self.alias_roles = [
            roles.get(
                "activation" if a["role"] == "output" else a["role"], roles["other"]
            )
            if self.attributed
            else roles["other"]
            for a in data.aliases
        ]
        self.seed = np.zeros((len(ROLE_GROUPS), len(data.edges)), dtype=np.int64)
        if self.attributed:
            accounting = data.pressure_accounting
            resident = set(accounting["resident_aliases"])
            final_device = set(accounting["final_device_aliases"])
            headroom = int(data.trace.header[2]) in (0, 1)
            spans = [[] for _ in data.aliases]
            for alias, first, last in data.trace.seed:
                spans[int(alias)].append((int(first), int(last)))
            for alias, intervals in enumerate(spans):
                if alias in resident:
                    continue
                charged = np.zeros(len(data.edges), dtype=bool)
                for first, last in intervals:
                    start = first - int(
                        headroom
                        and first > 0
                        and first - 1 not in data.productions[alias]
                    )
                    end = last - int(
                        last > 0
                        and alias not in final_device
                        and last not in data.input_uses[alias]
                    )
                    if end >= start:
                        charged[start : end + 1] = True
                for task in data.productions[alias] - data.input_uses[alias]:
                    charged[task] = True
                self.seed[self.alias_roles[alias], charged] += int(
                    data.trace.alias_sizes[alias]
                )
        else:
            # Older recordings lack the eligibility/final-location metadata.
            # Preserve the exact total rather than guessing its composition.
            self.seed[roles["other"]] = data.trace.initial[:, 1]
        if not np.array_equal(self.seed.sum(axis=0), data.trace.initial[:, 1]):
            raise ValueError(
                "role seed charges do not reconcile with recorded pressure"
            )
        self.resident_roles = np.zeros(len(ROLE_GROUPS), dtype=np.int64)
        accounting = data.pressure_accounting or {}
        self.resident_slice = accounting.get("resident_slice_bytes", 0)
        if "resident_reservations" in accounting:
            for alias, _generations, size in accounting["resident_reservations"]:
                self.resident_roles[self.alias_roles[alias]] += size
        else:
            self.resident_roles[roles["other"]] = self.resident_slice
        if int(self.resident_roles.sum()) != self.resident_slice:
            raise ValueError("role reservations do not reconcile with resident slice")
        self.offset = self.resident_slice if include_resident else 0
        self.workspace = np.asarray(
            [t["workspace_bytes"] for t in data.tasks] + [0], dtype=np.int64
        )
        capacity = data.trace.initial[:, 0].astype(np.int64).copy()
        limits = [
            (
                int((capacity + self.workspace).min()),
                int((capacity + self.workspace).max()),
            )
        ]
        for reduction in data.trace.reductions:
            capacity[reduction.changes[:, 0].astype(np.intp)] = reduction.changes[:, 1]
            limits.append(
                (
                    int((capacity + self.workspace).min()),
                    int((capacity + self.workspace).max()),
                )
            )
        maximum = max(high for _, high in limits)
        self.flat_budget = maximum + self.offset
        self.capacity_reserved = any(low < maximum for low, _ in limits)
        self.rid = None
        self.count = 0
        self.operation = 0
        self.values = self.seed.copy()

    def at(self, reduction, cut_count, pressure):
        """Return role-by-boundary bytes for this exact cut count."""
        if self.rid != reduction or cut_count < self.count:
            self.rid = reduction
            self.count = self.operation = 0
            self.values = self.seed.copy()
            for attempt in self.data.trace.reductions[: reduction + 1]:
                positions = attempt.changes[:, 0].astype(np.intp)
                self.values[-1, positions] = attempt.changes[:, 2]
        attempt = self.data.trace.reductions[reduction]
        while self.operation < len(attempt.pressure_ops):
            index, first, last, sign = (
                int(x) for x in attempt.pressure_ops[self.operation]
            )
            if index >= cut_count:
                break
            alias = int(attempt.cuts[index, 4])
            size = int(self.data.trace.alias_sizes[alias])
            self.values[self.alias_roles[alias], first : last + 1] += (
                size if sign == 1 else -size
            )
            self.operation += 1
        self.count = cut_count
        if np.any(self.values < 0) or not np.array_equal(
            self.values.sum(axis=0), pressure
        ):
            raise ValueError("role pressure does not reconcile with recorded total")
        return (
            self.values + self.resident_roles[:, None]
            if self.include_resident
            else self.values
        )

    def legend(self):
        """Keep the legend stable even after a role's current charge reaches zero."""
        return [
            (
                i,
                "Unattributed" if name == "other" and not self.attributed else label,
                color,
            )
            for i, (name, label, color) in enumerate(ROLE_GROUPS)
            if self.seed[i].any()
            or (self.include_resident and self.resident_roles[i] > 0)
            or (name == "workspace" and self.workspace.any())
            or (
                name == "repair"
                and (
                    self.capacity_reserved
                    or any(r.changes[:, 2].any() for r in self.data.trace.reductions)
                )
            )
        ]

    def display(self, reduction, cut_count, pressure, capacity):
        """Move workspace/reserved capacity into the stack beneath a flat budget."""
        values = self.at(reduction, cut_count, pressure).copy()
        values[-2] = self.workspace
        held = (
            self.flat_budget - capacity.astype(np.int64) - self.offset - self.workspace
        )
        if np.any(held < 0):
            raise ValueError("display budget is below a recorded capacity")
        values[-1] += held
        if not np.array_equal(
            values.sum(axis=0) - self.flat_budget,
            pressure.astype(np.int64) - capacity.astype(np.int64),
        ):
            raise ValueError("display composition changed a boundary's excess")
        return values
