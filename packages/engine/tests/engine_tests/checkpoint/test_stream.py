"""The existing rank archive streams to borrowed sinks and persistent artifacts."""

import hashlib
import io
import socket
import subprocess
import sys
import tarfile
from concurrent.futures import ThreadPoolExecutor
from threading import enumerate as threads

import pytest
import torch

from engine_tests.execution.test_checkpoint import MemoryExecutor
from engine_tests.server.test_artifacts import store
from shadowspill.checkpoint import (
    checkpoint_size,
    load_checkpoint,
    save_checkpoint,
    write_checkpoint,
)
from shadowspill.checkpoint.archive import WriteAll, read_manifest
from shadowspill.orchestration.launch.publication import OutputSpec
from shadowspill_client.protocol.requests import ArtifactRef


def test_partial_nonseekable_writes_match_atomic_file_and_freeze_metadata(tmp_path):
    executor = MemoryExecutor()
    metadata = {"updates": 7, "cursor": [1, 2]}
    path = tmp_path / "expected.ssp"
    save_checkpoint(executor, path, metadata=metadata, chunk_bytes=16)

    class ShortSink:
        def __init__(self):
            self.data = bytearray()

        def write(self, data):
            metadata["cursor"].append(3)
            count = min(17, len(data))
            self.data.extend(data[:count])
            return count

    sink = ShortSink()
    manifest = write_checkpoint(executor, sink, metadata=metadata, chunk_bytes=16)
    assert bytes(sink.data) == path.read_bytes()
    assert manifest["metadata"] == {"updates": 7, "cursor": [1, 2]}
    with tarfile.open(fileobj=io.BytesIO(sink.data), mode="r:") as archive:
        assert read_manifest(archive) == manifest


@pytest.mark.parametrize("count", [0, -1, None, True, 1000000])
def test_nonprogress_or_invalid_write_counts_fail(count):
    class InvalidSink:
        def write(self, data):
            return count

    with pytest.raises(OSError, match="valid write progress"):
        write_checkpoint(MemoryExecutor(), InvalidSink())


def test_large_sink_failure_joins_hash_work_before_returning():
    executor = MemoryExecutor()
    executor.values = {"large": torch.ones(1 << 20)}

    class FailedSink:
        def write(self, data):
            if len(data) >= 1 << 20:
                raise OSError("injected checkpoint sink failure")
            return len(data)

    before = {
        thread for thread in threads() if thread.name.startswith("checkpoint-hash")
    }
    with pytest.raises(OSError, match="sink failure"):
        write_checkpoint(executor, FailedSink(), chunk_bytes=1 << 20)
    assert {
        thread for thread in threads() if thread.name.startswith("checkpoint-hash")
    } == before


@pytest.mark.parametrize("fail", [False, True])
def test_socket_stream_commits_only_after_writer_success_and_restores(tmp_path, fail):
    executor = MemoryExecutor()
    expected = {key: value.clone() for key, value in executor.values.items()}
    spec = OutputSpec(
        "rank-0",
        checkpoint_size(executor, metadata={"updates": 9}),
        "application/octet-stream",
    )
    writer_socket, reader_socket = socket.socketpair()
    writer_socket.settimeout(5)
    reader_socket.settimeout(5)

    class NetworkSink:
        def __init__(self):
            self.digest = hashlib.sha256()

        def write(self, data):
            count = writer_socket.send(data)
            self.digest.update(data[:count])
            return count

    def export():
        try:
            sink = NetworkSink()
            result = write_checkpoint(
                executor, sink, metadata={"updates": 9}, chunk_bytes=16
            )
            if fail:
                raise OSError("rank failed before confirming completion")
            return result, sink.digest.hexdigest()
        finally:
            writer_socket.shutdown(socket.SHUT_WR)

    with (
        writer_socket,
        reader_socket,
        ThreadPoolExecutor(max_workers=1) as pool,
        store(tmp_path) as artifacts,
    ):
        pending = pool.submit(export)

        def receive(destination):
            while block := reader_socket.recv(64 << 10):
                WriteAll(destination).write(block)
            manifest, checksum = pending.result(5)
            assert manifest["logical_rank"] == 0
            return ArtifactRef(spec.id, checksum, spec.size_bytes, spec.media_type)

        if fail:
            with pytest.raises(OSError, match="before confirming"):
                artifacts.produce(spec, receive)
            assert artifacts.used == (0, 0)
        else:
            ref = artifacts.produce(spec, receive)
            path = tmp_path / "restore.ssp"
            with artifacts.open(ref) as source, path.open("wb") as destination:
                while block := source.read(64 << 10):
                    destination.write(block)
            for value in executor.values.values():
                value.zero_()
            assert load_checkpoint(executor, path, chunk_bytes=24) == {"updates": 9}
            for key, value in executor.values.items():
                torch.testing.assert_close(value.float(), expected[key].float())


def test_archive_inspection_imports_without_frameworks():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'torch', 'triton', 'pynvml', 'shadowspill_workloads'}:
            raise AssertionError('framework imported for archive metadata: ' + name)
sys.meta_path.insert(0, Block())
from shadowspill.checkpoint.archive import read_manifest, WriteAll
assert 'shadowspill.checkpoint.save' not in sys.modules
assert 'shadowspill.checkpoint.load' not in sys.modules
""",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stdout + result.stderr
