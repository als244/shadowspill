"""A certificate in its own file reads back as the certificate it was."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from shadowspill.ir import (
    AliasGroupSpec,
    MemoryAction,
    MemoryActionKind,
    MemoryLocation,
    MemorySchedule,
    ObjectSpec,
    ResidencySpec,
    ShadowSpillProgram,
    TaskProfile,
    TaskSpec,
)
from shadowspill.planner import (
    AdmissionFacts,
    CandidateDiagnostic,
    GenericPlanningOptions,
    PlanningDiagnostics,
    ProgramPlanResult,
    ResolvedProgramDiagnostics,
    TaskAdmissionSpec,
    TaskAllocationStep,
    TaskAllocationStepKind,
)
from shadowspill.planner.admission.layout.build import build_fixed_layout_admission
from shadowspill.planner.admission.layout.model import FixedLayoutAdmission
from shadowspill.planner.plan_store.reader import read_stored_selection
from shadowspill.planner.search import SearchOptions
from shadowspill.planner.serialization.certificate_file import (
    CERTIFICATE_FILE,
    read_certificate,
    read_certificate_meta,
    write_certificate,
)
from shadowspill.planner.serialization.simulation_result import simulation_result_value
from shadowspill.simulator import SimulationConfig, simulate

from ._examples import COMPUTE, DEVICE


def _selected(
    program: ShadowSpillProgram, schedule: MemorySchedule, config: SimulationConfig
) -> ProgramPlanResult:
    simulation = simulate(program, schedule, config=config)
    return ProgramPlanResult(
        program=program,
        search_options=SearchOptions(
            generic=GenericPlanningOptions(minimum_object_bytes_evict_eligible=0)
        ),
        initial_residency=schedule.initial_residency,
        final_residency=schedule.final_residency,
        simulation_config=config,
        schedule=schedule,
        selections=(),
        simulation=simulation,
        diagnostics=PlanningDiagnostics(
            selected_candidate_id="fixture",
            selected_selection_id="fixture",
            selected_makespan_ns=simulation.makespan_ns,
            resolved_programs=(
                ResolvedProgramDiagnostics(
                    selection_id="fixture",
                    choices=(),
                    selected_candidate_id="fixture",
                    selected_makespan_ns=simulation.makespan_ns,
                    candidate_evaluations=(
                        CandidateDiagnostic(
                            candidate_id="fixture",
                            selection_id="fixture",
                            status="valid",
                            makespan_ns=simulation.makespan_ns,
                        ),
                    ),
                ),
            ),
        ),
    )


def _admitted(scratch_bytes: int) -> FixedLayoutAdmission:
    return _case(scratch_bytes)[1]


def _case(scratch_bytes: int) -> tuple[ProgramPlanResult, FixedLayoutAdmission]:
    """An alias evicted and fetched back into the bytes it left: one reuse
    dependency. With scratch, the fetching task also allocates and releases
    a workspace, which fills the task-allocation columns."""

    capacity = 64 + scratch_bytes
    program = ShadowSpillProgram(
        devices=(DEVICE,),
        alias_groups=(AliasGroupSpec("state", "cuda_0", 64, retain_spill_copy=True),),
        objects=(ObjectSpec("state_object", "state", 0, 64),),
        profiles=(
            TaskProfile("profile", 10, 0, "abi"),
            TaskProfile("scratch", 10, scratch_bytes, "abi"),
        ),
        tasks=(
            TaskSpec("produce", COMPUTE, "profile", inputs=("state_object",)),
            TaskSpec("trigger", COMPUTE, "scratch", dependencies=("produce",)),
            TaskSpec(
                "consume",
                COMPUTE,
                "profile",
                dependencies=("trigger",),
                inputs=("state_object",),
            ),
        ),
    )
    schedule = MemorySchedule(
        initial_residency=(ResidencySpec("state", MemoryLocation.DEVICE),),
        actions=(
            MemoryAction("produce", "state", MemoryActionKind.EVICT),
            MemoryAction("trigger", "state", MemoryActionKind.FETCH),
        ),
        final_residency=(ResidencySpec("state", MemoryLocation.DEVICE),),
    )
    config = SimulationConfig.single_device(
        "cuda_0",
        device_capacity_bytes=capacity,
        spill_capacity_bytes=128,
        fetch_solo_bandwidth_bytes_per_second=64_000_000_000,
        fetch_concurrent_bandwidth_bytes_per_second=64_000_000_000,
        evict_solo_bandwidth_bytes_per_second=64_000_000_000,
        evict_concurrent_bandwidth_bytes_per_second=64_000_000_000,
    )
    steps = (
        (
            TaskAllocationStep(0, TaskAllocationStepKind.ALLOCATE, scratch_bytes),
            TaskAllocationStep(0, TaskAllocationStepKind.RELEASE),
        )
        if scratch_bytes
        else ()
    )
    facts = AdmissionFacts(
        "cuda_0",
        capacity,
        capacity,
        1,
        tuple(
            TaskAdmissionSpec(
                task.task_id,
                workspace_extents=(scratch_bytes,) if task.task_id == "trigger" else (),
                allocation_steps=steps if task.task_id == "trigger" else (),
            )
            if scratch_bytes
            else TaskAdmissionSpec(task.task_id)
            for task in program.tasks
        ),
    )
    selected = _selected(program, schedule, config)
    return selected, build_fixed_layout_admission(selected, facts)


CERTIFICATES = {
    "evicted_and_reused": lambda: _admitted(0),
    "with_scratch": lambda: _admitted(32),
}


@pytest.mark.parametrize("name", sorted(CERTIFICATES))
def test_a_certificate_reads_back_as_written(tmp_path: Path, name: str) -> None:
    admitted = CERTIFICATES[name]()
    assert admitted.simulation.task_intervals
    assert admitted.simulation.transfer_intervals
    path = tmp_path / CERTIFICATE_FILE
    write_certificate(path, admitted)

    again = read_certificate(path)

    assert again == admitted
    assert again.simulation.task_intervals == admitted.simulation.task_intervals
    assert again.simulation.transfer_intervals == admitted.simulation.transfer_intervals
    assert len(again.layout.reuse_dependencies) == 1
    if name == "with_scratch":
        assert len(again.layout.task_allocation_leases) == 1


@pytest.mark.parametrize("name", sorted(CERTIFICATES))
def test_a_certificate_read_back_writes_again_unchanged(
    tmp_path: Path, name: str
) -> None:
    """A read-back simulation holds records and no simulator columns, so
    writing it encodes them from the records: what reads back is the same."""

    admitted = CERTIFICATES[name]()
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    write_certificate(first, admitted)
    read_back = read_certificate(first)
    assert read_back.simulation.interval_arrays is None
    write_certificate(second, read_back)

    assert read_certificate(second) == admitted


def test_a_certificates_header_reads_without_its_arrays(tmp_path: Path) -> None:
    admitted = _admitted(32)
    path = tmp_path / CERTIFICATE_FILE
    write_certificate(path, admitted)

    meta = read_certificate_meta(path)

    assert meta["facts_digest"] == admitted.layout.facts_digest
    assert meta["layout"] == json.loads(json.dumps(admitted.layout.header()))
    assert "placements" not in meta["layout"]


def test_a_stored_record_reads_with_its_certificate_merged(tmp_path: Path) -> None:
    admitted = _admitted(32)
    record = tmp_path / "selection.json"
    record.write_text(json.dumps({"schema": "record"}))
    assert read_stored_selection(record) == {"schema": "record"}
    write_certificate(tmp_path / CERTIFICATE_FILE, admitted)

    whole = read_stored_selection(record)
    header = read_stored_selection(record, layout=False)

    certificate = whole["admission_certificate"]
    assert certificate["layout"] == admitted.layout.to_dict()
    assert certificate["facts_digest"] == admitted.layout.facts_digest
    assert len(certificate["simulation"]["transfer_intervals"]) == len(
        admitted.simulation.transfer_intervals
    )
    assert header["admission_certificate"] == {
        "facts_digest": admitted.layout.facts_digest,
        "layout": json.loads(json.dumps(admitted.layout.header())),
    }


def _rewritten(path: Path, change) -> None:
    with np.load(path, allow_pickle=False) as data:
        members = {name: data[name] for name in data.files}
    change(members)
    np.savez(path, **members)


def test_a_certificate_of_another_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / CERTIFICATE_FILE
    write_certificate(path, _admitted(32))

    def other_schema(members: dict[str, np.ndarray]) -> None:
        meta = json.loads(members["meta"].tobytes())
        meta["schema"] = "shadowspill.other/v1"
        members["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)

    _rewritten(path, other_schema)
    with pytest.raises(ValueError, match="schema"):
        read_certificate(path)
    with pytest.raises(ValueError, match="schema"):
        read_certificate_meta(path)


def test_interval_columns_of_unequal_length_are_refused(tmp_path: Path) -> None:
    path = tmp_path / CERTIFICATE_FILE
    write_certificate(path, _admitted(32))

    def shorter(members: dict[str, np.ndarray]) -> None:
        members["simulation.transfers.bytes"] = members["simulation.transfers.bytes"][
            :-1
        ]

    _rewritten(path, shorter)
    with pytest.raises(ValueError, match="transfers columns differ in length"):
        read_certificate(path)


def test_the_occupancy_walk_reads_a_certificate_from_its_file(tmp_path: Path) -> None:
    """The walk over a record whose certificate is a file beside it is the
    walk over the same record with the certificate written into it."""

    from shadowspill.diagnostics.occupancy import attribute

    selected, admitted = _case(32)
    record = {
        "program_digest": selected.program.digest,
        "schedule": selected.schedule.to_dict(),
        "selections": [],
        "simulation_result": simulation_result_value(selected.simulation),
    }
    program = json.loads(json.dumps(selected.program.to_dict()))
    beside = tmp_path / "beside"
    beside.mkdir()
    (beside / "selection.json").write_text(json.dumps(record))
    write_certificate(beside / CERTIFICATE_FILE, admitted)
    within = json.loads(
        json.dumps(
            {**record, "admission_certificate": {"layout": admitted.layout.to_dict()}}
        )
    )

    from_file = attribute(read_stored_selection(beside / "selection.json"), program)
    from_record = attribute(within, program)

    assert from_file.execution is not None and from_record.execution is not None
    assert from_file.execution.intervals == from_record.execution.intervals
    assert from_file.execution.required_bytes == admitted.layout.required_bytes
    assert len(from_file.execution.intervals) == len(admitted.layout.placements)
    assert from_file.spill.intervals == from_record.spill.intervals


def test_a_simulation_column_of_another_type_is_refused(tmp_path: Path) -> None:
    path = tmp_path / CERTIFICATE_FILE
    write_certificate(path, _admitted(32))

    def floating(members: dict[str, np.ndarray]) -> None:
        members["simulation.tasks.start_ns"] = members[
            "simulation.tasks.start_ns"
        ].astype(np.float64)

    _rewritten(path, floating)
    with pytest.raises(ValueError, match=r"simulation column tasks\.start_ns"):
        read_certificate(path)


def test_a_simulation_index_past_its_names_is_refused(tmp_path: Path) -> None:
    path = tmp_path / CERTIFICATE_FILE
    write_certificate(path, _admitted(32))

    def past(members: dict[str, np.ndarray]) -> None:
        aliases = members["simulation.transfers.alias"].copy()
        aliases[0] = 1_000
        members["simulation.transfers.alias"] = aliases

    _rewritten(path, past)
    with pytest.raises(ValueError, match="index or code nothing names"):
        read_certificate(path)
