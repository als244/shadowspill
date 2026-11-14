"""Actual inherited private sockets, startup/execute fencing and owned cleanup."""

import os
import socket
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from engine_tests.training.test_worker_loop import call
from engine_tests.training.test_worker_startup import description
from shadowspill.orchestration.launch import connected
from shadowspill.orchestration.launch.connected import ConnectedProcess
from shadowspill.orchestration.launch.supervision import OwnedProcess
from shadowspill.training.worker.startup import WorkerStartupConnection, start_workers
from shadowspill_client.protocol.codec import ProtocolError


def launch(root, mode="normal", timeout=5):
    return ConnectedProcess(
        lambda fd: [
            sys.executable,
            str(Path(__file__).with_name("_connected_worker.py")),
            str(fd),
            mode,
            str(root / "effects"),
        ],
        environment={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        directory=root / "logs",
        cwd=root,
        log_limit_bytes=1024,
        startup_timeout=timeout,
    )


def startup(child):
    endpoint = WorkerStartupConnection(
        replace(description(0), logical_by_worker=(0,)), child.stream, child.stream
    )
    return start_workers(
        {0: endpoint},
        stop=lambda: child.close(grace_seconds=0.1, terminate_seconds=0.1),
    )


def test_private_control_survives_large_logs_and_closes_orderly_on_eof(tmp_path):
    with launch(tmp_path) as child:
        with startup(child) as group:
            child.ready()
            assert child._socket.gettimeout() is None
            with pytest.raises(RuntimeError, match="already ready"):
                child.ready()
            replies = group.run({0: call()})
            assert replies[0].status == "completed"
            assert (tmp_path / "effects").read_text() == "initialized\n1\n"
        assert child.closed and child.process.returncode == 0
    assert (tmp_path / "effects").read_text() == "initialized\n1\nclosed\n"
    assert child.stream.closed
    assert all(
        log.bytes_seen == 1 << 20 and log.truncated for log in child.process.logs
    )
    assert all(log.bytes_written == 1024 for log in child.process.logs)


def test_process_death_before_ready_never_returns_live_connections(tmp_path):
    with launch(tmp_path, "lost_ready") as child:
        with pytest.raises(ProtocolError, match="evidence"):
            startup(child)
        assert child.closed and child.stream.closed
        assert child.process.returncode == 42
    assert (tmp_path / "effects").read_text() == "initialized\n"


def test_lost_execute_reply_stops_owned_process_and_never_replays(tmp_path):
    with launch(tmp_path, "lost_completion") as child, startup(child) as group:
        child.ready()
        with pytest.raises(ProtocolError, match="absent"):
            group.run({0: call()})
        assert not group.valid and child.closed and child.process.returncode == 42
        with pytest.raises(RuntimeError, match="never replay"):
            group.run({0: call(2, 2)})
    assert (tmp_path / "effects").read_text() == "initialized\n1\n"


def test_startup_io_timeout_closes_socket_and_terminates_only_the_owned_child(tmp_path):
    with launch(tmp_path, "stall", timeout=0.05) as child:
        with pytest.raises(TimeoutError):
            startup(child)
        assert child.closed and child.stream.closed
        assert child.process.returncode is not None
    assert not (tmp_path / "effects").exists()


@pytest.mark.parametrize("fds", [(0,), (2,), (True,), (3, 3), tuple(range(3, 36))])
def test_invalid_inherited_descriptor_set_is_rejected_before_process_or_logs(
    tmp_path, fds
):
    with pytest.raises(ValueError, match="distinct private descriptors"):
        OwnedProcess(
            [sys.executable, "-c", "pass"],
            environment=os.environ,
            directory=tmp_path / "logs",
            cwd=tmp_path,
            pass_fds=fds,
        )
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize("failure", ["command", "spawn", "timeout"])
def test_failed_construction_closes_both_socket_ends(tmp_path, monkeypatch, failure):
    endpoints = socket.socketpair()
    monkeypatch.setattr(connected, "socketpair", lambda: endpoints)

    def command(fd):
        if failure == "command":
            raise ValueError("command construction failed")
        return ["/no/such/worker"]

    with pytest.raises((ValueError, OSError, OverflowError)):
        ConnectedProcess(
            command,
            environment=os.environ,
            directory=tmp_path / "logs",
            cwd=tmp_path,
            startup_timeout=1e300 if failure == "timeout" else 5,
        )
    assert all(endpoint.fileno() == -1 for endpoint in endpoints)
