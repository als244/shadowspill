"""Two real workers stream one admitted persistent generation through the group."""

import hashlib
import subprocess
import sys
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from engine_tests.server.test_artifacts import store
from engine_tests.training.test_worker_group import commands
from engine_tests.training.test_worker_loop import SCOPE
from engine_tests.training.test_worker_stream import SIZE
from shadowspill.orchestration.launch.publication import OutputSpec
from shadowspill.training.session.dispatch import ExecutionNotStarted
from shadowspill.training.worker.connection import WorkerConnection
from shadowspill.training.worker.group import WorkerGroup
from shadowspill_client.protocol.codec import ProtocolError


@contextmanager
def processes(root, *, failure=False):
    children, connections = [], {}

    def stop():
        for process in children:
            if process.poll() is None:
                process.kill()
        for process in children:
            process.wait(timeout=10)

    try:
        for rank in range(2):
            child = subprocess.Popen(
                [
                    sys.executable,
                    str(Path(__file__).with_name("_worker_stream.py")),
                    str(root / f"executions-{rank}"),
                    "lost_completion" if failure and rank == 1 else "normal",
                    str(rank),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
            )
            children.append(child)
            connections[rank] = WorkerConnection(
                replace(SCOPE, rank=rank), child.stdout, child.stdin
            )
        with WorkerGroup(connections, stop=stop) as group:
            yield group
    finally:
        stop()
        for process in children:
            for stream in (process.stdin, process.stdout, process.stderr):
                stream.close()


def snapshot(group, artifacts, invocation=1):
    with group.staged(commands(invocation, invocation)) as ready:
        specs = [
            OutputSpec(f"output-{rank}-{invocation}", SIZE, "application/octet-stream")
            for rank in group.connections
        ]
        ranks = {spec.id: rank for rank, spec in enumerate(specs)}

        def produce(spec, writer):
            if not ready.released:
                ready.execute()  # Group reservation has succeeded before this callback.
            rank = ranks[spec.id]
            ready.connections[rank].receive_stream(writer, size_bytes=SIZE)
            return ready.finish(rank).result

        return artifacts.stage_group(specs, produce)


def test_two_rank_streams_remain_private_until_the_owner_commits(tmp_path):
    with processes(tmp_path) as group, store(tmp_path) as artifacts:
        with snapshot(group, artifacts) as staged:
            assert not artifacts.page() and artifacts.used == (2 * SIZE, 2)
            assert group.valid
            refs = staged.commit()
        assert {ref.id for ref in refs} == {"output-0-1", "output-1-1"}
        for ref in refs:
            with artifacts.open(ref) as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == ref.sha256
        assert [(tmp_path / f"executions-{rank}").read_text() for rank in range(2)] == [
            "1\n",
            "1\n",
        ]


def test_storage_rejection_aborts_real_workers_then_corrected_call_runs_once(tmp_path):
    with processes(tmp_path) as group:
        with store(tmp_path, max_bytes=2 * SIZE - 1) as artifacts:
            with pytest.raises(ExecutionNotStarted, match="capacity"):
                snapshot(group, artifacts)
            assert artifacts.used == (0, 0)
            assert not list(tmp_path.glob("executions-*"))
            assert group.valid and not group.closed
        with store(tmp_path) as artifacts, snapshot(group, artifacts, 2) as staged:
            staged.commit()
        assert [(tmp_path / f"executions-{rank}").read_text() for rank in range(2)] == [
            "2\n",
            "2\n",
        ]


def test_lost_second_completion_discards_both_files_and_stops_the_group(tmp_path):
    with processes(tmp_path, failure=True) as group, store(tmp_path) as artifacts:
        with pytest.raises(ProtocolError):
            snapshot(group, artifacts)
        assert artifacts.used == (0, 0) and not artifacts.page()
        assert not list(artifacts._scratch.iterdir())
        assert group.closed and not group.valid
        assert [(tmp_path / f"executions-{rank}").read_text() for rank in range(2)] == [
            "1\n",
            "1\n",
        ]
