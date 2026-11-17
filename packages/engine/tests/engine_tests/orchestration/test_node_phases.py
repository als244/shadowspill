"""Real bounded child processes share allocation accounting with persistent workers."""

import json
import os
import sys
from contextlib import ExitStack
from dataclasses import replace
from threading import Event, Thread
from time import monotonic

import pytest

from shadowspill.orchestration.launch.admission import (
    AssignedDevice,
    NodeAdmission,
    NodeCapacity,
    ReservationCancelled,
    ResourceRequest,
    ResourcesUnavailable,
)
from shadowspill.orchestration.launch.host_arena import HostArena
from shadowspill.orchestration.launch.phases import NodePhases, PhaseInvocation
from shadowspill.training.preparation.phases.launcher import run_phases
from shadowspill_client.buffers import BufferBudget


def resources():
    return NodeAdmission(
        NodeCapacity(
            "node",
            (
                AssignedDevice("a", "GPU-a", 1024),
                AssignedDevice("b", "GPU-b", 1024),
            ),
            64 << 20,
        )
    )


def runner(node, arena, disk, root, *, devices=("a",), **options):
    return NodePhases(
        node,
        devices=devices,
        host_bytes=8 << 20,
        root=root,
        disk=disk,
        memory=arena.budget,
        log_limit_bytes=1024,
        **options,
    )


def invocation(root, phase="build", *, code=None, workers=1):
    code = code or "import os; print(os.environ['CUDA_VISIBLE_DEVICES'])"
    return PhaseInvocation(
        phase,
        (sys.executable, "-c", code),
        {**os.environ, "CUDA_VISIBLE_DEVICES": "unassigned"},
        workers,
        root,
        root / (phase + ".log"),
    )


def await_file(path):
    deadline = monotonic() + 5
    while not path.exists():
        assert monotonic() < deadline, "phase did not reach its handshake"
        Event().wait(0.005)


def test_gpu_phase_slot_order_and_cpu_planning_release_children_but_keep_logs(tmp_path):
    node, disk = resources(), BufferBudget(1 << 20, 10)
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    with HostArena(node, 1 << 20, 10) as arena:
        with runner(node, arena, disk, tmp_path, devices=("b", "a")) as launch:
            first = launch(invocation(tmp_path, workers=2))
            assert (
                first.returncode == 0 and first.logs[0].read_text() == "GPU-b,GPU-a\n"
            )
            assert not node.used.devices and node.used.host_bytes == 1 << 20
            with node.reserve(ResourceRequest({"a": 1024, "b": 1024}, 0)):
                planned = launch(invocation(tmp_path, "plan"))
                assert planned.logs[0].read_text() == "\n"
                assert node.used.devices == {"a", "b"}
            assert disk.used[1] == arena.budget.used[1] == 2
            for logs in launch.logs.values():
                references = logs.references()
                with logs.open(references["stdout"]) as stream:
                    assert stream.read() in (b"GPU-b,GPU-a\n", b"\n")
        assert disk.used == arena.budget.used == (0, 0)
    assert node.used.claims == 0 and os.environ.get("CUDA_VISIBLE_DEVICES") == original


@pytest.mark.parametrize("fail_progress", [False, True])
def test_admitted_progress_and_actual_phase_use_one_complete_host_claim(
    tmp_path, fail_progress
):
    node, disk, progress = resources(), BufferBudget(1 << 20, 10), []
    with HostArena(node, 9 << 20, 2) as arena:
        scratch = arena.partition(1 << 20, 10)
        processes = arena.partition(8 << 20, 1)

        def admitted(phase):
            assert node.used.host_bytes == 9 << 20
            assert node.used.devices == {"a"} and processes.budget.used == (8 << 20, 1)
            progress.append(phase)
            if fail_progress:
                raise OSError("cannot publish progress")

        with runner(
            node, scratch, disk, tmp_path, host=processes, admitted=admitted
        ) as launch:
            with (
                node.reserve(ResourceRequest({"a": 1024}, 0)),
                pytest.raises(ResourcesUnavailable),
            ):
                launch(invocation(tmp_path))
            assert not progress and processes.budget.used == (0, 0)
            if fail_progress:
                with pytest.raises(OSError, match="publish progress"):
                    launch(invocation(tmp_path))
            else:
                assert launch(invocation(tmp_path)).returncode == 0
            assert progress == ["build"] and processes.budget.used == (0, 0)
            assert node.used.claims == 1
        scratch.close()
        processes.close()
    assert node.used.claims == 0 and disk.used == (0, 0)


def test_disjoint_phase_jobs_use_separate_environment_snapshots_concurrently(tmp_path):
    node, disk = resources(), BufferBudget(1 << 20, 10)
    outcomes, errors, threads = [], [], []
    with HostArena(node, 1 << 20, 10) as arena, ExitStack() as stack:
        for device in ("a", "b"):
            launch = stack.enter_context(
                runner(node, arena, disk, tmp_path, devices=(device,))
            )
            ready, release = tmp_path / (device + ".ready"), tmp_path / "release"
            code = f"""
import os, time
from pathlib import Path
Path({str(ready)!r}).write_text(os.environ['CUDA_VISIBLE_DEVICES'])
while not Path({str(release)!r}).exists():
    time.sleep(0.005)
print(os.environ['CUDA_VISIBLE_DEVICES'])
"""

            command = invocation(tmp_path, code=code)

            def run(owner=launch, command=command):
                try:
                    outcomes.append(owner(command))
                except BaseException as error:
                    errors.append(error)

            thread = Thread(target=run)
            thread.start()
            threads.append(thread)
        try:
            await_file(tmp_path / "a.ready")
            await_file(tmp_path / "b.ready")
            assert node.used.devices == {"a", "b"}
            assert (tmp_path / "a.ready").read_text() == "GPU-a"
            assert (tmp_path / "b.ready").read_text() == "GPU-b"
        finally:
            (tmp_path / "release").touch()
            for thread in threads:
                thread.join(timeout=5)
                assert not thread.is_alive()
        assert not errors and len(outcomes) == 2
    assert node.used.claims == 0 and disk.used == (0, 0)


def test_busy_or_impossible_phases_fail_before_process_or_log_creation(tmp_path):
    node, disk = resources(), BufferBudget(1 << 20, 10)
    with (
        HostArena(node, 1 << 20, 10) as arena,
        runner(node, arena, disk, tmp_path) as launch,
    ):
        with (
            node.reserve(ResourceRequest({"a": 1024}, 0)),
            pytest.raises(ResourcesUnavailable),
        ):
            launch(invocation(tmp_path))
        with pytest.raises(ValueError, match="participants"):
            launch(invocation(tmp_path, workers=2))
        assert not launch.logs and not list(tmp_path.iterdir())
        assert disk.used == arena.budget.used == (0, 0)
    assert node.used.claims == 0


def test_stop_terminates_the_owned_running_phase_and_keeps_final_logs(tmp_path):
    node, disk = resources(), BufferBudget(1 << 20, 10)
    errors = []
    ready = tmp_path / "ready"
    code = f"""
from pathlib import Path
import time
Path({str(ready)!r}).touch()
print('running', flush=True)
while True:
    time.sleep(1)
"""
    with HostArena(node, 1 << 20, 10) as arena:
        launch = runner(node, arena, disk, tmp_path)

        def run():
            try:
                launch(invocation(tmp_path, code=code))
            except BaseException as error:
                errors.append(error)

        thread = Thread(target=run)
        thread.start()
        try:
            await_file(ready)
            launch.stop()
            thread.join(timeout=5)
            assert not thread.is_alive() and len(errors) == 1
            assert isinstance(errors[0], ReservationCancelled)
            assert not node.used.devices and disk.used[1] == 1
            logs = launch.logs["build"]
            assert logs.directory.joinpath("stdout.log").read_text() == "running\n"
        finally:
            launch.stop()
            thread.join(timeout=5)
            launch.close()
    assert node.used.claims == 0 and disk.used == (0, 0)


def test_phase_log_deletion_failure_remains_owned_and_retryable(tmp_path, monkeypatch):
    node, disk = resources(), BufferBudget(1 << 20, 10)
    with HostArena(node, 1 << 20, 10) as arena:
        launch = runner(node, arena, disk, tmp_path)
        launch(invocation(tmp_path))
        temporary = launch.logs["build"]._temporary
        cleanup = temporary.cleanup
        before = disk.used, arena.budget.used

        def fail():
            raise OSError("cleanup unavailable")

        monkeypatch.setattr(temporary, "cleanup", fail)
        try:
            with pytest.raises(BaseExceptionGroup, match="logs"):
                launch.close()
            assert not launch.closed and (disk.used, arena.budget.used) == before
        finally:
            monkeypatch.setattr(temporary, "cleanup", cleanup)
            launch.close()
    assert node.used.claims == 0 and disk.used == (0, 0)


def test_stopping_a_waiting_job_does_not_close_the_shared_allocation(tmp_path):
    node, disk = resources(), BufferBudget(1 << 20, 10)
    errors, entered = [], Event()
    with (
        HostArena(node, 1 << 20, 10) as arena,
        runner(node, arena, disk, tmp_path, wait_for_resources=True) as launch,
        node.reserve(ResourceRequest({"a": 1024}, 0)) as borrowed,
    ):

        def waiting():
            entered.set()
            try:
                launch(invocation(tmp_path))
            except BaseException as error:
                errors.append(error)

        thread = Thread(target=waiting)
        thread.start()
        assert entered.wait(5)
        # Wait until the job owns its launch attempt, then stop it.
        deadline = monotonic() + 5
        while not launch._active:
            assert monotonic() < deadline
            Event().wait(0.005)
        launch.stop()
        thread.join(timeout=2)
        assert not thread.is_alive() and len(errors) == 1
        assert isinstance(errors[0], ReservationCancelled)
        assert not borrowed.closed and not node.closed
        assert node.used.devices == {"a"} and not launch.logs
    assert node.used.claims == 0


def test_failed_spawn_releases_node_slots_but_retains_owned_log_files(tmp_path):
    node, disk = resources(), BufferBudget(1 << 20, 10)
    with (
        HostArena(node, 1 << 20, 10) as arena,
        runner(node, arena, disk, tmp_path) as launch,
    ):
        command = replace(invocation(tmp_path), command=(str(tmp_path / "missing"),))
        with pytest.raises(FileNotFoundError):
            launch(command)
        assert not node.used.devices and node.used.host_bytes == 1 << 20
        assert disk.used[1] == arena.budget.used[1] == 1
        assert all(
            reference.size_bytes == 0
            for reference in launch.logs["build"].references().values()
        )
    assert node.used.claims == 0 and disk.used == (0, 0)


def test_phase_logs_are_capped_and_report_truncation_without_blocking_child(tmp_path):
    node, disk = resources(), BufferBudget(1 << 20, 10)
    with (
        HostArena(node, 1 << 20, 10) as arena,
        runner(node, arena, disk, tmp_path) as launch,
    ):
        command = invocation(
            tmp_path,
            code="import os; os.write(1, b'x' * (1 << 20)); os.write(2, b'done\\n')",
        )
        result = launch(command)
        assert result.returncode == 0 and result.log_statuses[0].truncated
        assert result.logs[0].stat().st_size == 1024
        assert result.logs[1].read_text() == "done\n"
    assert disk.used == (0, 0) and node.used.claims == 0


def test_failed_termination_retains_the_whole_lease_until_explicit_retry(
    tmp_path, monkeypatch
):
    node, disk = resources(), BufferBudget(1 << 20, 10)
    with HostArena(node, 1 << 20, 10) as arena:
        launch = runner(node, arena, disk, tmp_path)
        launch(invocation(tmp_path))
        # A later failed group cleanup is tested at the actual lease boundary.
        lease = node.reserve(ResourceRequest({"a": 1024}, 8 << 20))
        child = lease.launch_command(
            (sys.executable, "-c", "pass"),
            devices=("a",),
            environment=os.environ,
            directory=tmp_path / "retry",
            cwd=tmp_path,
        )
        child.wait(timeout=5)
        close = child.close

        def fail():
            raise OSError("termination not confirmed")

        monkeypatch.setattr(child, "close", fail)
        try:
            with pytest.raises(BaseExceptionGroup, match="cleanup"):
                lease.close()
            assert not lease.closed and node.used.devices == {"a"}
        finally:
            monkeypatch.setattr(child, "close", close)
            lease.close()
            launch.close()
    assert node.used.claims == 0


def test_existing_phase_driver_uses_owned_launches_and_stops_after_failure(
    tmp_path, monkeypatch
):
    from shadowspill.training.preparation.phases import launcher as phases

    node, disk = resources(), BufferBudget(1 << 20, 10)
    commands = []

    def command(module, args, *, ranks):
        commands.append(module)
        code = (
            "import sys; print('diagnostic', file=sys.stderr); sys.exit(7)"
            if module.endswith("profile")
            else "pass"
        )
        return [sys.executable, "-c", code]

    monkeypatch.setattr(phases, "worker_command", command)
    with (
        HostArena(node, 1 << 20, 10) as arena,
        runner(node, arena, disk, tmp_path) as launch,
    ):
        with pytest.raises(RuntimeError, match="profile failed with status 7") as error:
            run_phases(
                "installed",
                tmp_path,
                "all",
                phases=("build", "profile", "plan"),
                launch=launch,
            )
        assert "diagnostic" in str(error.value)
        assert commands == ["installed.build", "installed.profile"]
        rows = [
            json.loads(line)
            for line in (tmp_path / "logs/profile-launcher.jsonl")
            .read_text()
            .splitlines()
        ]
        returned = next(row for row in rows if row.get("name") == "workers.returned")
        assert returned["returncode"] == 7 and len(returned["log_statuses"]) == 2
        assert not node.used.devices
    assert node.used.claims == 0
