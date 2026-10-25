"""Read and replay the single-device PressureFit trace, one frame at a time.

Positions in the binary file are zero-based cells: position 0 precedes task 0,
and position p > 0 follows task p - 1. Cut endpoints use the algorithm's
boundary numbers instead (position minus one). See docs/python/pressurefit-trace.md.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .occupancy import ROLE_PRIORITY

Words = NDArray[np.uint64]
Spans = tuple[tuple[tuple[int, int], ...], ...]
_MutableFrame = tuple[int, list[list[tuple[int, int]]], Words, Words]
_WIDTHS = (8, 1, 3, 2, 1, 3, 96, 4, 7, 2, 7, 3, 9)
_NONE = (1 << 64) - 1
_CRITERIA = (
    "exposed_ns",
    "writeback",
    "negative_slack_ns",
    "negative_size_bytes",
    "alias",
    "start",
    "cut_id",
)


def _digest(values: Words) -> int:
    result = 14695981039346656037
    for value in values.astype("<u8", copy=False).tobytes():
        result = ((result ^ value) * 1099511628211) & _NONE
    return result


def _span_digest(spans: Spans) -> int:
    words = []
    for alias in spans:
        words.append(len(alias))
        for first, last in alias:
            words.extend((first, last))
    return _digest(np.asarray(words, dtype="<u8"))


def _cut(spans: list[list[tuple[int, int]]], choice: Words) -> None:
    alias = int(choice[1])
    start, end = (int(word) for word in choice[2:4].view("<i8"))
    updated: list[tuple[int, int]] = []
    for first, last in spans[alias]:
        if start > end:
            split = end + 1
            if first <= split < last:
                updated.extend(((first, split), (split + 1, last)))
            else:
                updated.append((first, last))
        elif last < start + 1 or first > end + 1:
            updated.append((first, last))
        else:
            if first < start + 1:
                updated.append((first, start))
            if last > end + 1:
                updated.append((end + 2, last))
    spans[alias] = updated


@dataclass(frozen=True)
class TraceState:
    """A snapshot after cut_count cuts in one reduction (zero is the seed)."""

    reduction: int
    cut_count: int
    spans: Spans
    pressure: Words
    capacity: Words


@dataclass(frozen=True)
class TraceReduction:
    """Array blocks for one seed-to-residency attempt; arrays are read-only."""

    changes: Words
    cuts: Words
    pressure_ops: Words
    outcome: Words
    reason: Words | None = None

    @property
    def status(self) -> int:
        return int(self.outcome[1])


@dataclass(frozen=True)
class PressureFitTrace:
    """One candidate's complete reductions and explicit answer reference."""

    header: Words
    alias_sizes: Words
    seed: Words
    initial: Words
    reductions: tuple[TraceReduction, ...]
    step_reductions: Words
    steps: Words
    outcome: Words

    @classmethod
    def read(cls, path: str | Path) -> PressureFitTrace:
        """Decode bounded array blocks; reject incomplete or unknown formats."""
        blocks: list[tuple[int, Words]] = []
        widths = list(_WIDTHS)
        path = Path(path)
        size = path.stat().st_size
        with path.open("rb") as stream:
            if stream.read(8) != b"PFTRACE\0":
                raise ValueError("invalid PressureFit trace magic")
            while stream.tell() < size:
                raw = stream.read(16)
                if len(raw) != 16:
                    raise ValueError("truncated trace block header")
                kind, width, rows = struct.unpack("<IIQ", raw)
                if kind >= len(widths) or width != widths[kind]:
                    raise ValueError("unknown trace block kind or width")
                length = rows * width * 8
                if length > size - stream.tell():
                    raise ValueError("truncated trace block data")
                data = np.frombuffer(stream.read(length), dtype="<u8")
                blocks.append((kind, data.reshape(rows, width)))
        cursor = 0

        def take(kind: int, rows: int | None = None) -> Words:
            nonlocal cursor
            if cursor >= len(blocks) or blocks[cursor][0] != kind:
                raise ValueError(f"missing or out-of-order trace block {kind}")
            data = blocks[cursor][1]
            cursor += 1
            if rows is not None and len(data) != rows:
                raise ValueError(f"incorrect row count for trace block {kind}")
            return data

        header = take(0, 1)[0]
        if int(header[0]) not in (3, 4):
            raise ValueError("unsupported PressureFit trace version")
        if int(header[4]) != 0 or int(header[7]) != 2:
            raise ValueError("trace does not use the supported PressureFit comparator")
        alias_sizes = take(1, int(header[5]))[:, 0]
        seed = take(2)
        initial = take(3, int(header[6]))
        reductions: list[TraceReduction] = []
        links = []
        while cursor < len(blocks) and blocks[cursor][0] in (4, 9):
            if blocks[cursor][0] == 9:
                links.append(take(9, 1)[0])
                continue
            index = len(reductions)
            if int(take(4, 1)[0, 0]) != index:
                raise ValueError("nonsequential reduction ID")
            reason = take(12, 1)[0] if int(header[0]) >= 4 else None
            changes = take(5)
            chunks: dict[int, list[Words]] = {6: [], 7: []}
            while cursor < len(blocks) and blocks[cursor][0] in chunks:
                kind = blocks[cursor][0]
                chunks[kind].append(take(kind))

            def join(kind: int, chunks: dict[int, list[Words]] = chunks) -> Words:
                array = (
                    np.concatenate(chunks[kind])
                    if chunks[kind]
                    else np.empty((0, widths[kind]), dtype="<u8")
                )
                array.flags.writeable = False
                return array

            outcome = take(8, 1)[0]
            if int(outcome[0]) != index:
                raise ValueError("mismatched reduction end")
            reductions.append(
                TraceReduction(changes, join(6), join(7), outcome, reason)
            )
        steps = take(10)
        outcome = take(11, 1)[0]
        if cursor != len(blocks):
            raise ValueError("unexpected data after trace footer")
        step_reductions = np.asarray(links, dtype="<u8").reshape(-1, 2)
        step_reductions.flags.writeable = False
        trace = cls(
            header,
            alias_sizes,
            seed,
            initial,
            tuple(reductions),
            step_reductions,
            steps,
            outcome,
        )
        trace._validate()
        return trace

    @property
    def cut_score(self) -> str:
        return "split-exposed"

    @property
    def cut_ranking(self) -> str:
        return "slack-after-writeback"

    @property
    def ranking_criteria(self) -> tuple[str, ...]:
        return _CRITERIA[int(self.header[2]) in (1, 3) :]

    def ranking_key(self, reduction: int, cut: int, rank: int = 0) -> tuple[int, ...]:
        """Exact comparator key, including the final deterministic cut ID."""
        row = self.reductions[reduction].cuts[cut]
        if not 0 <= rank <= int(row[2]):
            raise IndexError("cut alternative rank is out of range")
        choice = row[3 + 12 * rank : 15 + 12 * rank]
        evict, fetch, se, sf = map(int, row[66 + 8 * rank : 70 + 8 * rank])
        signed = choice.view("<i8")
        values = (
            int(choice[4]),
            int(signed[5]),
            evict + fetch - se - sf,
            int(signed[8]),
            int(signed[10]),
            int(signed[11]),
            int(choice[0]),
        )
        return values[int(self.header[2]) in (1, 3) :]

    def _validate(self) -> None:
        count = len(self.initial)
        if count == 0 or not self.reductions:
            raise ValueError("trace has no boundaries or reductions")
        if len(self.seed) and (
            np.any(self.seed[:, 0] >= len(self.alias_sizes))
            or np.any(self.seed[:, 1] > self.seed[:, 2])
            or np.any(self.seed[:, 2] >= count)
        ):
            raise ValueError("invalid seed spans")
        if len(self.steps) != len(self.step_reductions) or any(
            int(link[0]) != index or int(link[1]) >= len(self.reductions)
            for index, link in enumerate(self.step_reductions)
        ):
            raise ValueError("invalid step-to-reduction links")
        if self.answer_step is not None and self.answer_step >= len(self.steps):
            raise ValueError("invalid answer step")
        if self.answer_step is not None and (
            not int(self.steps[self.answer_step, 3]) & 32
            or self.steps[self.answer_step, 0] != self.outcome[2]
        ):
            raise ValueError("answer step does not match the candidate outcome")
        for reduction in self.reductions:
            if reduction.reason is not None and (
                int(reduction.reason[0]) > 4 or int(reduction.reason[4]) > 1
            ):
                raise ValueError("invalid reduction reason or pool")
            cuts, ops, changes = (
                reduction.cuts,
                reduction.pressure_ops,
                reduction.changes,
            )
            if len(changes) and np.any(changes[:, 0] >= count):
                raise ValueError("invalid capacity/extra-pressure position")
            if len(cuts) and (
                np.any(cuts[:, 0] >= count)
                or np.any(cuts[:, 2] > 3)
                or np.any(cuts[:, 51] < 1)
                or np.any(cuts[:, 51] > 3)
            ):
                raise ValueError("invalid cut trigger or alternative count")
            for row in cuts:
                choices = row[3:51].reshape(4, 12)[: int(row[2]) + 1]
                if np.any(choices[:, 1] >= len(self.alias_sizes)):
                    raise ValueError("invalid cut alias")
                endpoints = choices[:, 2:4].view("<i8")
                if np.any(endpoints < -1) or np.any(endpoints >= count):
                    raise ValueError("invalid cut endpoints")
                for rank, choice in enumerate(choices):
                    _, _, evict, fetch, evict_slack, fetch_slack, ee, ef = (
                        int(v) for v in row[64 + rank * 8 : 72 + rank * 8]
                    )
                    expected = (
                        min(_NONE, ee + ef) if int(self.header[2]) not in (1, 3) else 0
                    )
                    if (
                        ee != max(0, evict - evict_slack)
                        or ef != max(0, fetch - fetch_slack)
                        or int(choice[4]) != expected
                    ):
                        raise ValueError("inconsistent split score components")
            if len(ops) and (
                np.any(ops[:, 0] >= len(cuts))
                or np.any(ops[1:, 0] < ops[:-1, 0])
                or np.any(ops[:, 1] > ops[:, 2])
                or np.any(ops[:, 2] >= count)
                or np.any((ops[:, 3] != 1) & (ops[:, 3] != _NONE))
            ):
                raise ValueError("invalid pressure operations")

    @property
    def answer_step(self) -> int | None:
        value = int(self.outcome[1])
        return None if value == _NONE else value

    @property
    def answer_reduction(self) -> int | None:
        step = self.answer_step
        return None if step is None else int(self.step_reductions[step, 1])

    def object_capacity_bytes(self, reduction: int) -> int | None:
        """Object budget from this pass's search steps, or an unchanged prior one.

        Repair-only passes retain the budget. An unlinked capacity refinement
        or a pass with no recorded reason cannot safely inherit it.
        An unspecified placement budget is encoded as zero.
        """
        if not 0 <= reduction < len(self.reductions):
            raise IndexError("reduction index out of range")
        by_reduction: dict[int, int] = {}
        for step, rid in self.step_reductions:
            value = int(self.steps[int(step), 1])
            rid = int(rid)
            if rid in by_reduction and by_reduction[rid] != value:
                raise ValueError("object capacity changed within one reduction")
            by_reduction[rid] = value
        capacity = None
        for rid, attempt in enumerate(self.reductions[: reduction + 1]):
            if rid in by_reduction:
                capacity = by_reduction[rid] or None
            elif attempt.reason is None or int(attempt.reason[0]) not in (1, 2, 4):
                capacity = None
        return capacity

    def reduction_reason(self, reduction: int) -> dict[str, Any] | None:
        """The recorded allocation/placement that prompted this pass; v3 lacks it."""
        row = self.reductions[reduction].reason
        if row is None:
            return None
        kind, status, task, alias, location, requested, used, capacity, shortfall = map(
            int, row
        )
        kinds = (
            "initial_seed",
            "simulation_pressure",
            "admission_pressure",
            "capacity_refinement",
            "escalation_rollback",
        )
        if kind >= len(kinds):
            raise ValueError("unknown reduction reason")
        return {
            "kind": kinds[kind],
            "status": status,
            "task_position": task if task < len(self.initial) - 1 else None,
            "alias_index": alias if alias < len(self.alias_sizes) else None,
            "pool": "object"
            if kind == 3
            else "execution"
            if location == 0
            else "spill",
            "requested_bytes": requested,
            "used_bytes": used,
            "capacity_bytes": capacity,
            "shortfall_bytes": shortfall,
        }

    def _walk(self, reduction: int) -> Iterator[_MutableFrame]:
        if not 0 <= reduction < len(self.reductions):
            raise IndexError("reduction index out of range")
        capacity = self.initial[:, 0].copy()
        extra = np.zeros_like(capacity)
        for attempt in self.reductions[: reduction + 1]:
            positions = attempt.changes[:, 0].astype(np.intp)
            capacity[positions] = attempt.changes[:, 1]
            extra[positions] = attempt.changes[:, 2]
        capacity.flags.writeable = False
        pressure = self.initial[:, 1] + extra
        spans: list[list[tuple[int, int]]] = [[] for _ in self.alias_sizes]
        for alias, first, last in self.seed:
            spans[int(alias)].append((int(first), int(last)))

        yield 0, spans, pressure, capacity
        attempt = self.reductions[reduction]
        operation = 0
        for index, row in enumerate(attempt.cuts):
            choice = row[3:15]
            _cut(spans, choice)
            size = int(self.alias_sizes[int(choice[1])])
            while operation < len(attempt.pressure_ops):
                cut, first, last, sign = (
                    int(v) for v in attempt.pressure_ops[operation]
                )
                if cut != index:
                    break
                pressure[first : last + 1] += np.uint64(
                    size if sign == 1 else (-size & _NONE)
                )
                operation += 1
            yield index + 1, spans, pressure, capacity

    @staticmethod
    def _snapshot(reduction: int, frame: _MutableFrame) -> TraceState:
        cuts, spans, pressure, capacity = frame
        copy = pressure.copy()
        copy.flags.writeable = False
        return TraceState(
            reduction, cuts, tuple(tuple(item) for item in spans), copy, capacity
        )

    def iter_states(self, reduction: int) -> Iterator[TraceState]:
        """Yield seed and successive cuts; retain no previous frame internally."""
        for frame in self._walk(reduction):
            yield self._snapshot(reduction, frame)

    def state(self, reduction: int, cut_count: int | None = None) -> TraceState:
        """Replay a requested frame; None means the reduction's last frame."""
        if not 0 <= reduction < len(self.reductions):
            raise IndexError("reduction index out of range")
        total = len(self.reductions[reduction].cuts)
        cut_count = total if cut_count is None else cut_count
        if not 0 <= cut_count <= total:
            raise IndexError("cut count out of range")
        for frame in self._walk(reduction):
            if frame[0] == cut_count:
                return self._snapshot(reduction, frame)
        raise AssertionError("unreachable frame")

    def comparisons(self, reduction: int, cut: int) -> tuple[dict[str, Any], ...]:
        """Explain the first ranking field separating each alternative."""
        row = self.reductions[reduction].cuts[cut]
        choices = row[3:51].reshape(4, 12)[: int(row[2]) + 1]
        chosen = self.ranking_key(reduction, cut)
        result = []
        previous = chosen
        for rank, choice in enumerate(choices[1:], 1):
            alternative = self.ranking_key(reduction, cut, rank)
            if alternative <= previous:
                raise ValueError("trace alternatives are not strictly ranked")
            field = next(
                i
                for i, (a, b) in enumerate(zip(chosen, alternative, strict=True))
                if a != b
            )
            result.append(
                {
                    "cut_id": int(choice[0]),
                    "criterion": self.ranking_criteria[field],
                    "chosen_value": chosen[field],
                    "alternative_value": alternative[field],
                }
            )
            previous = alternative
        return tuple(result)

    def verify(self) -> None:
        """Check every trigger, top-three boundary ranking and final C digest."""
        for index, reduction in enumerate(self.reductions):
            frames = self._walk(index)
            _, spans, pressure, capacity = next(frames)
            for cut, row in enumerate(reduction.cuts):
                over = np.flatnonzero(pressure > capacity)
                # Stable ordering preserves the earlier-position tie break.
                # Subtract from UINT64_MAX instead of negating unsigned bytes.
                order = np.argsort(
                    _NONE - (pressure[over] - capacity[over]), kind="stable"
                )
                ranked = [int(p) for p in over[order[:3]]]
                actual = row[52:64].reshape(3, 4)[: int(row[51])]
                expected = np.asarray(
                    [
                        (
                            p,
                            int(pressure[p]),
                            int(capacity[p]),
                            int(pressure[p]) - int(capacity[p]),
                        )
                        for p in ranked
                    ],
                    dtype="<u8",
                ).reshape(-1, 4)
                if (
                    not np.array_equal(actual, expected)
                    or not ranked
                    or (
                        int(row[0]) != ranked[0]
                        or int(row[1]) != int(pressure[ranked[0]])
                    )
                ):
                    raise ValueError(
                        f"boundary ranking mismatch at reduction {index}, cut {cut}"
                    )
                self.comparisons(index, cut)
                _, spans, pressure, capacity = next(frames)
            if _span_digest(tuple(tuple(item) for item in spans)) != int(
                reduction.outcome[5]
            ):
                raise ValueError(f"residency digest mismatch at reduction {index}")
            if _digest(pressure) != int(reduction.outcome[6]):
                raise ValueError(f"pressure digest mismatch at reduction {index}")


def boundary_labels(
    tasks: Sequence[Mapping[str, Any]],
    task_metadata: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Join labels by task ID, preserving both neighbours of every boundary.

    Optional metadata may supply label, phase, stage_index and microbatch.
    Indices are displayed as supplied; no stage or batch is inferred from IDs.
    """
    names = []
    for task in tasks:
        task_id = str(task["task_id"])
        metadata = dict((task_metadata or {}).get(task_id, {}))
        phase = metadata.get("phase", task.get("phase"))
        parts = [str(phase)] if phase else []
        if metadata.get("stage_index") is not None:
            parts.append(f"stage {metadata['stage_index']}")
        if metadata.get("microbatch") is not None:
            parts.append(f"microbatch {metadata['microbatch']}")
        label = metadata.get("label") or ", ".join(parts)
        if len(parts) <= 1 and "label" not in metadata:
            label = f"{label} ({task_id})" if label else task_id
        names.append({**dict(task), **metadata, "task_id": task_id, "label": label})
    return [
        {
            "position": p,
            "boundary": p - 1,
            "after": names[p - 1] if p else None,
            "before": names[p] if p < len(names) else None,
        }
        for p in range(len(names) + 1)
    ]


def stage_timings(
    tasks: Sequence[Mapping[str, Any]],
    task_metadata: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Group plan durations by semantic phase/stage/repetition, preserving gaps."""
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for task in tasks:
        metadata = task_metadata.get(str(task["task_id"]), {})
        stage = metadata.get("stage_index")
        if stage is None:
            continue
        key = (
            metadata.get("phase", task.get("phase")),
            stage,
            metadata.get("microbatch"),
        )
        groups.setdefault(key, []).append(task)
    result = []
    for (phase, stage, microbatch), members in groups.items():
        timed = all(member.get("plan_start_ns") is not None for member in members)
        start = (
            min(int(member["plan_start_ns"]) for member in members) if timed else None
        )
        end = max(int(member["plan_end_ns"]) for member in members) if timed else None
        result.append(
            {
                "phase": phase,
                "stage_index": stage,
                "microbatch": microbatch,
                "task_ids": [member["task_id"] for member in members],
                "profiled_duration_ns": sum(
                    int(member["profiled_duration_ns"]) for member in members
                ),
                "plan_duration_ns": sum(
                    int(member["plan_duration_ns"]) for member in members
                )
                if timed
                else None,
                "plan_start_ns": start,
                "plan_end_ns": end,
                "plan_span_ns": end - start
                if end is not None and start is not None
                else None,
            }
        )
    return result


def read_manifest(
    directory: str | Path,
    *,
    semantic_labels: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read a completed recording manifest; partial reruns are not evidence."""
    value: dict[str, Any] = json.loads((Path(directory) / "manifest.json").read_text())
    if value.get("schema") != "shadowspill.pressurefit_trace/v1":
        raise ValueError("unsupported trace manifest")
    if value.get("status") != "complete":
        raise ValueError("trace recording did not complete")
    sidecar = Path(directory) / "semantic_labels.json"
    if semantic_labels is None and sidecar.is_file():
        semantic_labels = json.loads(sidecar.read_text())
    if (
        semantic_labels
        and semantic_labels.get("program_digest") != value["program_digest"]
    ):
        raise ValueError("semantic labels belong to a different program")
    semantic_labels = semantic_labels or {}
    task_metadata = semantic_labels.get("tasks", {})
    object_metadata = semantic_labels.get("objects", {})
    labelled_contexts: set[int] = set()
    for candidate in value["candidates"]:
        context_index = candidate.get("context_index")
        if context_index is not None:
            contexts = value.get("contexts", ())
            if (
                not isinstance(context_index, int)
                or not 0 <= context_index < len(contexts)
                or contexts[context_index]["selection_id"] != candidate["selection_id"]
            ):
                raise ValueError("invalid candidate context")
            context = contexts[context_index]
            candidate["aliases"] = context["aliases"]
            candidate["task_ids"] = context["task_ids"]
        candidate["boundaries"] = boundary_labels(candidate["tasks"], task_metadata)
        candidate["stages"] = stage_timings(candidate["tasks"], task_metadata)
        if context_index is not None and context_index in labelled_contexts:
            continue
        if context_index is not None:
            labelled_contexts.add(context_index)
        task_names = {
            row["before"]["task_id"]: row["before"]["label"]
            for row in candidate["boundaries"]
            if row["before"]
        }
        for alias in candidate["aliases"]:
            for member in alias["objects"]:
                metadata = object_metadata.get(member["object_id"], {})
                member["semantic_metadata"] = metadata
                label = metadata.get("label")
                if not label:
                    producer = member["producer"]
                    origins = [
                        item
                        for item in metadata.get("origins", ())
                        if item["task_id"] == producer
                    ]
                    if origins:
                        label = (
                            f"{member['category']}, "
                            f"output {origins[0]['output_index']} "
                            f"of {task_names[producer]}"
                        )
                    elif producer in task_names:
                        label = f"{member['category']} from {task_names[producer]}"
                member["label"] = (
                    label or f"{member['category']}: {member['object_id']}"
                )
            if alias["objects"]:
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
    return value
