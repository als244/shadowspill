"""Tracing is observational, and its streamed cuts reproduce the C state."""

from __future__ import annotations

import json
import struct
from dataclasses import replace
from pathlib import Path

import pytest

from reference.python.pressurefit.facts import build_facts
from reference.python.pressurefit.residency import (
    ResidencyPlan,
    Span,
    _cut_score,
    legal_cuts,
)
from shadowspill.diagnostics.pressurefit_trace import (
    PressureFitTrace,
    boundary_labels,
    read_manifest,
    stage_timings,
)
from shadowspill.ir import AliasGroupSpec, ObjectSpec, TaskSpec
from shadowspill.planner import GenericPlanningOptions, SearchOptions, plan_program
from shadowspill.planner.search.algorithms.pressurefit import PressureFit
from shadowspill.planner.search.algorithms.pressurefit.options import PressureFitOptions
from shadowspill.planner.search.algorithms.pressurefit.search import (
    build_problems,
    run_problems,
)
from shadowspill.planner.search.algorithms.pressurefit.tracing import record_plan_trace

from ...._examples import (
    COMPUTE,
    training_chain_config,
    training_chain_initial,
    training_chain_program,
)
from ....test_annotated_plan import _pressurefit_program


def _record(
    tmp_path: Path,
    layers: int = 10,
    capacity: int = 500,
):
    algorithm = PressureFit(
        PressureFitOptions(
            residency_strategies=(
                "headroom-stall",
                "headroom-transfer",
                "tight-stall",
                "tight-transfer",
            ),
            fetch_rules=("packed-fifo", "packed-fit", "latest-safe", "demand"),
        )
    )
    options = SearchOptions(
        generic=GenericPlanningOptions(
            minimum_object_bytes_evict_eligible=0, deterministic=True
        ),
        algorithm=algorithm,
        workers=1,
    )
    result = algorithm(
        training_chain_program(layers),
        initial_residency=training_chain_initial(layers),
        config=training_chain_config(capacity),
        generic=options.generic,
        workers=1,
    )
    pairs = tuple(
        (parent.selection_id, candidate.candidate_id)
        for parent in result.diagnostics.resolved_programs
        for candidate in parent.candidate_evaluations
    )
    directory = record_plan_trace(
        result,
        options=replace(
            options,
            trace_directory=str(tmp_path),
            trace_candidates=pairs,
        ),
    )
    return result, directory


@pytest.mark.parametrize(
    ("layers", "capacity"), [(1, 224), (3, 256), (10, 232), (10, 500)]
)
def test_every_policy_replays_without_changing_its_answer(tmp_path, layers, capacity):
    result, directory = _record(tmp_path, layers, capacity)
    raw = json.loads((directory / "manifest.json").read_text())
    assert len(raw["contexts"]) == 1
    assert all("aliases" not in item for item in raw["candidates"])
    manifest = read_manifest(directory)
    assert manifest["answer"]["schedule_digest"] == result.schedule.digest
    assert len(manifest["candidates"]) == 16
    assert all(
        item["aliases"] is manifest["contexts"][0]["aliases"]
        for item in manifest["candidates"]
    )
    for candidate in manifest["candidates"]:
        trace = PressureFitTrace.read(directory / candidate["file"])
        trace.verify()
        assert trace.header[0] == 4
        assert trace.reduction_reason(0)["kind"] == "initial_seed"
        for rid in range(1, len(trace.reductions)):
            assert trace.reduction_reason(rid)["kind"] != "initial_seed"
        assert candidate["verified_against_original"]
        assert candidate["resolution_recompute_fraction"] == "0"
        assert len(candidate["boundaries"]) == len(trace.initial)
        assert len(candidate["aliases"]) == len(trace.alias_sizes)
        profiles = {p.profile_id: p.runtime_ns for p in result.program.profiles}
        for task in candidate["tasks"]:
            assert task["profiled_duration_ns"] == profiles[task["profile_id"]]
            if task["plan_start_ns"] is not None:
                assert (
                    task["plan_duration_ns"]
                    == task["plan_end_ns"] - task["plan_start_ns"]
                )
                assert (
                    task["plan_stall_ns"]
                    == task["plan_start_ns"] - task["plan_ready_ns"]
                )
        if trace.answer_step is not None:
            assert trace.steps[trace.answer_step, 0] == trace.outcome[2]
            assert trace.steps[trace.answer_step, 3] & 32  # SHADOWSPILL_STEP_ANSWER
        if (layers, capacity) == (10, 232) and candidate["candidate_id"] == (
            "tight-stall/packed-fit"
        ):
            # Fetch-only repairs eventually worsen this candidate. Its answer
            # restores the kept schedule, rather than the last attempted step.
            assert trace.answer_step == 0
            assert trace.answer_step < len(trace.steps) - 1
            assert trace.steps[trace.answer_step, 0] < trace.steps[-1, 0]
        for reduction in range(len(trace.reductions)):
            assert trace.state(reduction, 0).spans == trace.state(0, 0).spans


def test_normal_api_traces_cached_winner_without_changing_key(tmp_path):
    options = SearchOptions(
        generic=GenericPlanningOptions(deterministic=True), workers=1
    )
    plain = plan_program(
        _pressurefit_program(),
        search_options=options,
        artifact_store=tmp_path / "store",
        verbose=False,
    )
    traced_options = replace(options, trace_directory=str(tmp_path / "traces"))
    assert traced_options.to_dict() == options.to_dict()
    for _ in range(2):
        traced = plan_program(
            _pressurefit_program(),
            search_options=traced_options,
            artifact_store=tmp_path / "store",
            verbose=False,
        )
        assert traced.result.schedule == plain.result.schedule
        assert traced.result.diagnostics == plain.result.diagnostics
    manifests = list((tmp_path / "traces").glob("*/manifest.json"))
    assert len(manifests) == 2
    for path in manifests:
        value = read_manifest(path.parent)
        assert len(value["candidates"]) == 1
        assert (
            value["candidates"][0]["candidate_id"]
            == plain.result.diagnostics.selected_candidate_id
        )
        trace = PressureFitTrace.read(path.parent / value["candidates"][0]["file"])
        trace.verify()
        by_id = {t["task_id"]: t for t in value["candidates"][0]["tasks"]}
        for interval in plain.result.simulation.task_intervals:
            assert by_id[interval.task_id]["plan_start_ns"] == interval.start_ns
            assert by_id[interval.task_id]["plan_end_ns"] == interval.end_ns


def test_ranked_alternatives_match_exhaustive_reference(tmp_path):
    result, directory = _record(tmp_path, 3, 256)
    facts = build_facts(
        result.program, (), result.initial_residency, (), result.simulation_config
    )
    compared = 0
    for candidate in read_manifest(directory)["candidates"]:
        trace = PressureFitTrace.read(directory / candidate["file"])
        assert trace.cut_ranking == "slack-after-writeback"
        if int(trace.header[2]) not in (2, 3):
            continue
        score_kind = "min-transfer" if int(trace.header[2]) == 3 else "min-stall"
        for index, reduction in enumerate(trace.reductions):
            frames = trace.iter_states(index)
            for cut_index, (row, frame) in enumerate(
                zip(reduction.cuts, frames, strict=False)
            ):
                residency = ResidencyPlan(
                    tuple(
                        tuple(Span(a - 1, b - 1) for a, b in spans)
                        for spans in frame.spans
                    ),
                    facts.anchors,
                )
                eligible = legal_cuts(facts, residency, int(row[0]) - 1, "cuda_0")
                ranked = sorted(
                    eligible,
                    key=lambda cut: _cut_score(
                        facts,
                        result.simulation_config,
                        cut,
                        score_kind,
                        int(row[0]) - 1,
                    ),
                )[:4]
                actual = row[3:51].reshape(4, 12)[: int(row[2]) + 1]
                assert len(actual) == len(ranked)
                for rank, (choice, expected) in enumerate(
                    zip(actual, ranked, strict=True)
                ):
                    assert (int(choice[1]), *map(int, choice[2:4].view("<i8"))) == (
                        expected.alias,
                        expected.start,
                        expected.end,
                    )
                    assert trace.ranking_key(index, cut_index, rank)[:-1] == _cut_score(
                        facts,
                        result.simulation_config,
                        expected,
                        score_kind,
                        int(row[0]) - 1,
                    )
                    compared += 1
    assert compared > 20


def test_incomplete_and_corrupted_traces_are_rejected(tmp_path):
    _, directory = _record(tmp_path, 1, 224)
    path = next(directory.rglob("*.pftrace"))
    data = path.read_bytes()
    for length in (0, 12, len(data) - 1, len(data) - 40):
        broken = tmp_path / "broken.pftrace"
        broken.write_bytes(data[:length])
        with pytest.raises(ValueError):
            PressureFitTrace.read(broken)
    value = json.loads((directory / "manifest.json").read_text())
    value["status"] = "recording"
    (directory / "manifest.json").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="did not complete"):
        read_manifest(directory)


def test_v3_trace_keeps_replay_without_inventing_reduction_causes(tmp_path):
    _, directory = _record(tmp_path, 1, 224)
    raw = next(directory.rglob("*.pftrace")).read_bytes()
    legacy = bytearray(raw[:8])
    cursor = 8
    while cursor < len(raw):
        kind, width, rows = struct.unpack_from("<IIQ", raw, cursor)
        end = cursor + 16 + 8 * width * rows
        if kind != 12:
            chunk = bytearray(raw[cursor:end])
            if kind == 0:
                struct.pack_into("<Q", chunk, 16, 3)
            legacy.extend(chunk)
        cursor = end
    path = tmp_path / "v3.pftrace"
    path.write_bytes(legacy)
    trace = PressureFitTrace.read(path)
    trace.verify()
    assert trace.header[0] == 3
    assert all(
        trace.reduction_reason(rid) is None for rid in range(len(trace.reductions))
    )


def test_trace_io_failures_do_not_silently_lose_records(tmp_path):
    generic = GenericPlanningOptions(
        minimum_object_bytes_evict_eligible=0, deterministic=True
    )
    options = PressureFitOptions(
        residency_strategies=("tight-stall",),
        fetch_rules=("demand",),
    )
    problems = build_problems(
        training_chain_program(3),
        training_chain_initial(3),
        (),
        training_chain_config(256),
        None,
        resolutions=((),),
        progress=None,
    )
    for destination in (tmp_path / "missing",):
        with pytest.raises(RuntimeError):
            run_problems(
                problems, generic, options, workers=1, trace_directory=str(destination)
            )
    run_problems(problems, generic, options, workers=1, trace_directory=str(tmp_path))
    files = {path: path.read_bytes() for path in tmp_path.glob("*.pftrace")}
    with pytest.raises(RuntimeError):
        run_problems(
            problems, generic, options, workers=1, trace_directory=str(tmp_path)
        )
    assert all(path.read_bytes() == data for path, data in files.items())


def test_boundary_labels_join_by_task_id_and_keep_both_neighbours():
    tasks = [
        {"task_id": "f", "phase": "forward"},
        {"task_id": "b", "phase": "backward"},
    ]
    labels = boundary_labels(
        tasks,
        {
            "b": {"stage_index": 12, "microbatch": 3},
            "f": {"stage_index": 12, "microbatch": 3},
        },
    )
    assert labels[0]["after"] is None
    assert labels[-1]["before"] is None
    assert labels[1]["after"]["label"] == "forward, stage 12, microbatch 3"
    assert labels[1]["before"]["label"] == "backward, stage 12, microbatch 3"
    assert labels[1]["boundary"] == 0


def test_nondeterministic_trace_is_explicitly_refused(tmp_path):
    generic = GenericPlanningOptions(minimum_object_bytes_evict_eligible=0)
    result = PressureFit()(
        training_chain_program(1),
        initial_residency=training_chain_initial(1),
        config=training_chain_config(224),
        generic=generic,
        workers=1,
    )
    with pytest.raises(ValueError, match="deterministic=True"):
        record_plan_trace(
            result,
            options=SearchOptions(generic=generic, trace_directory=str(tmp_path)),
        )
    assert not list(tmp_path.iterdir())


def test_long_stream_keeps_more_cuts_than_aliases(tmp_path):
    original = training_chain_program(1)
    aliases = tuple(
        AliasGroupSpec(f"a{i}", "cuda_0", 64, retain_spill_copy=True) for i in range(3)
    )
    program = replace(
        original,
        alias_groups=aliases,
        objects=tuple(ObjectSpec(f"o{i}", f"a{i}", 0, 64) for i in range(3)),
        tasks=(
            original.tasks[0],
            *(
                TaskSpec(
                    f"t{i}",
                    COMPUTE,
                    "forward_profile",
                    inputs=(f"o{i % 3}",),
                    dependencies=((f"t{i - 1}" if i else "start"),),
                )
                for i in range(450)
            ),
        ),
    )
    from shadowspill.ir import MemoryLocation, ResidencySpec

    initial = tuple(
        ResidencySpec(a.alias_group_id, MemoryLocation.SPILL) for a in aliases
    )
    generic = GenericPlanningOptions(
        minimum_object_bytes_evict_eligible=0, deterministic=True
    )
    algorithm = PressureFit(
        PressureFitOptions(
            residency_strategies=("tight-transfer",),
            fetch_rules=("demand",),
        )
    )
    result = algorithm(
        program,
        initial_residency=initial,
        config=training_chain_config(128),
        generic=generic,
        workers=1,
    )
    directory = record_plan_trace(
        result,
        options=SearchOptions(
            generic=generic, algorithm=algorithm, trace_directory=str(tmp_path)
        ),
    )
    trace = PressureFitTrace.read(next(directory.rglob("*.pftrace")))
    assert len(trace.reductions[0].cuts) > 128  # Cross the streamed block boundary.
    assert len(trace.reductions[0].cuts) > len(trace.alias_sizes)
    trace.verify()


def test_semantic_object_names_do_not_change_recorded_ids(tmp_path):
    result, directory = _record(tmp_path, 1, 224)
    labels = {
        "program_digest": result.program.digest,
        "tasks": {"f_0": {"stage_index": 0, "microbatch": 4}},
        "objects": {
            "W_0": {
                "label": "parameter decoder.block.0.weight",
                "names": ["decoder.block.0.weight"],
            }
        },
    }
    manifest = read_manifest(directory, semantic_labels=labels)
    alias = next(
        item
        for item in manifest["candidates"][0]["aliases"]
        if item["alias_id"] == "W_0"
    )
    assert alias["label"] == "parameter decoder.block.0.weight"
    assert alias["objects"][0]["object_id"] == "W_0"
    assert (
        read_manifest(directory)["candidates"][0]["aliases"]
        != manifest["candidates"][0]["aliases"]
    )
    with pytest.raises(ValueError, match="different program"):
        read_manifest(directory, semantic_labels={**labels, "program_digest": "wrong"})


def test_stage_timing_distinguishes_compute_from_elapsed_span():
    tasks = [
        {
            "task_id": "first",
            "profiled_duration_ns": 10,
            "plan_duration_ns": 10,
            "plan_start_ns": 5,
            "plan_end_ns": 15,
        },
        {
            "task_id": "second",
            "profiled_duration_ns": 20,
            "plan_duration_ns": 20,
            "plan_start_ns": 25,
            "plan_end_ns": 45,
        },
    ]
    metadata = {
        name: {"phase": "forward", "stage_index": 2, "microbatch": 1}
        for name in ("second", "first")
    }
    (stage,) = stage_timings(tasks, metadata)
    assert stage["task_ids"] == ["first", "second"]
    assert stage["profiled_duration_ns"] == stage["plan_duration_ns"] == 30
    assert stage["plan_span_ns"] == 40
