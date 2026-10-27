"""Verified construction order and explanations of the fixed placement.

Replay uses the shipped C placer, checks every saved offset, and then walks
the exact occupied-address union only for the placements being explained.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from shadowspill.planner.admission.placement import place_lifetimes


def aligned(value, alignment):
    return -(-int(value) // int(alignment)) * int(alignment)


class PlacementReplay:
    def __init__(self, data):
        self.data = data
        self.leases = data.leases
        self.resident = np.array(
            [p["alias_group_id"] in data.resident_aliases for p in self.leases],
            dtype=bool,
        )
        self.starts = np.array(
            [p["predicted_start_ns"] for p in self.leases], dtype=np.int64
        )
        self.ends = np.array(
            [p["predicted_end_ns"] for p in self.leases], dtype=np.int64
        )
        self.sizes = np.array([p["bytes"] for p in self.leases], dtype=np.int64)
        self.offsets = np.array([p["offset"] for p in self.leases], dtype=np.int64)
        main = np.flatnonzero(~self.resident)
        times = np.unique(np.concatenate((self.starts[main], self.ends[main])))
        self.spans = np.searchsorted(times, self.ends) - np.searchsorted(
            times, self.starts
        )
        order = sorted(
            main.tolist(),
            key=lambda i: (-self.sizes[i], -self.spans[i], self.starts[i], i),
        )
        self.main_count = len(order)
        self.order = np.array(
            order + np.flatnonzero(self.resident).tolist(), dtype=np.int64
        )
        self.ranks = np.empty(len(self.order), dtype=np.int64)
        self.ranks[self.order] = np.arange(len(self.order))
        self.main_bytes = 0
        self.verified = False

    def verify(self):
        offsets, span = place_lifetimes(
            [SimpleNamespace(**p) for p in self.leases], self.resident.tolist()
        )
        offsets = list(offsets)
        self.main_bytes = span
        for i in np.flatnonzero(self.resident):
            offsets[i] = aligned(span, self.leases[i]["alignment"])
            span = offsets[i] + self.leases[i]["bytes"]
        if (
            offsets != self.offsets.tolist()
            or span != self.data.layout["fixed_slice_bytes"]
        ):
            raise ValueError("placement replay differs from saved offsets or span")
        self.verified = True
        return {
            "verified_offsets": len(offsets),
            "main_bytes": self.main_bytes,
            "fixed_slice_bytes": span,
            "resident_allocations": int(self.resident.sum()),
        }

    def detail(self, index):
        if not self.verified:
            raise ValueError("verify the replay before explaining placement")
        lease = self.leases[index]
        rank = int(self.ranks[index])
        prior = self.ranks < rank
        overlap = (
            (self.starts < self.ends[index])
            & (self.ends > self.starts[index])
            & (self.ends > self.starts)
        )
        if self.ends[index] <= self.starts[index]:
            overlap[:] = False
        blockers = np.flatnonzero(prior & overlap & ~self.resident)
        ranges = sorted(
            (int(self.offsets[i]), int(self.offsets[i] + self.sizes[i]), int(i))
            for i in blockers
        )
        union = []
        for lo, hi, _ in ranges:
            if union and lo <= union[-1][1]:
                union[-1][1] = max(union[-1][1], hi)
            else:
                union.append([lo, hi])
        cursor, attempts = 0, []
        if self.resident[index]:
            before = max(
                (
                    self.leases[i]["offset"] + self.leases[i]["bytes"]
                    for i in self.order[:rank]
                ),
                default=0,
            )
            cursor = aligned(before, lease["alignment"])
        else:
            for lo, hi in union:
                cursor = aligned(cursor, lease["alignment"])
                if cursor + lease["bytes"] <= lo:
                    break
                if cursor < hi:
                    attempts.append({"offset": cursor, "blocked_range": [lo, hi]})
                    cursor = hi
            cursor = aligned(cursor, lease["alignment"])
        if cursor != lease["offset"]:
            raise ValueError(
                f"explanation does not reproduce lease {lease['lease_id']}"
            )
        reusable = np.flatnonzero(
            prior
            & (self.ends <= self.starts[index])
            & (self.offsets < self.offsets[index] + self.sizes[index])
            & (self.offsets + self.sizes > self.offsets[index])
        )
        return {
            "lease_index": int(index),
            "lease_id": lease["lease_id"],
            "placement_number": rank + 1,
            "size_bytes": lease["bytes"],
            "alignment": lease["alignment"],
            "lifetime_boundary_span": int(self.spans[index]),
            "offset": cursor,
            "resident_slice": bool(self.resident[index]),
            "occupied_union": union,
            "attempts": attempts,
            "overlapping_lease_indices": blockers.tolist(),
            "reused_predecessor_indices": reusable.tolist(),
        }

    def examples(self, count=3):
        candidates = [
            int(i)
            for i in self.order[: self.main_count]
            if self.sizes[i] >= 2**20 and self.ends[i] > self.starts[i]
        ]
        if not candidates:
            candidates = [int(i) for i in self.order[: self.main_count]]
        chosen = []
        for i in candidates:
            if not chosen or (len(chosen) == 1 and self.offsets[i] > 0):
                chosen.append(i)
            elif len(chosen) >= 2 and self.leases[i]["purpose"] != "task_workspace":
                detail = self.detail(i)
                if detail["reused_predecessor_indices"]:
                    chosen.append(i)
            if len(chosen) >= count:
                break
        for i in candidates:
            if len(chosen) >= count:
                break
            if i not in chosen:
                chosen.append(i)
        return sorted(chosen, key=lambda i: self.ranks[i])
