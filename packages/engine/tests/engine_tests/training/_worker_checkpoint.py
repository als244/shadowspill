"""Owned CPU process exercising the actual checkpoint command/stream encoder."""

import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine_tests.execution.test_checkpoint import MemoryExecutor
from shadowspill.orchestration.launch.file_grants import resolve_file
from shadowspill.training.session.controls import CallControls
from shadowspill.training.worker.bindings import BoundWorkerCall
from shadowspill.training.worker.checkpoint_reader import WorkerCheckpointReader
from shadowspill.training.worker.checkpoint_writer import WorkerCheckpointWriter
from shadowspill.training.worker.execution import WorkerExecution
from shadowspill.training.worker.loop import WorkerLoop
from shadowspill.training.worker.protocol import WorkerScope
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef

rank, marker, mode = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
executor = MemoryExecutor()
executor.plan.program.digest = "a" * 64
executor.logical_rank = rank
executor.close = lambda: None
executor.discard_gradients = lambda: None
executor.synchronize = lambda: None
if mode == "load_failure":

    def fail_load(*args, **kwargs):
        executor.values["step"].zero_()
        raise OSError("restore failed after tensor mutation")

    executor.initialize = fail_load
scope = WorkerScope("run", "server", f"process-{rank}", rank)
budget = BufferBudget(64 << 10)
reader = None
if mode == "frozen":
    from engine_tests.training.test_worker_frozen_checkpoint import fixture

    executor, budget, base, reader, _, _ = fixture(marker.parent)
    executor.close = lambda: None
    reader.resolve = lambda ref: resolve_file(Path(sys.argv[4]), ref)
elif mode != "no_loader":
    reader = WorkerCheckpointReader(
        executor,
        budget,
        lambda ref: resolve_file(Path(sys.argv[4]), ref),
        chunk_bytes=4 if mode == "small_staging" else 32,
        world_size=int(sys.argv[5]),
    )
writer = WorkerCheckpointWriter(
    executor,
    budget,
    sys.stdout.buffer,
    plans=ArtifactRef("plans", hashlib.sha256(b"plans").hexdigest(), 5),
    chunk_bytes=32,
    frozen_base=lambda: None if reader is None else reader.frozen_base,
)
execution = WorkerExecution(
    executor,
    SimpleNamespace(scope=scope),
    budget,
    lambda *args: None,
    checkpoints=writer,
    restore=reader,
)


def bind(command):
    call = BoundWorkerCall(command, None, CallControls(), {}, ())
    if mode == "no_encoder":
        return call
    return execution.prepare(call)


def execute(bound):
    with marker.open("a") as output:
        output.write(f"{bound.command.invocation}\n")
        output.flush()
        os.fsync(output.fileno())
    result = execution(bound)
    if mode == "lost_completion" or (
        mode == "lost_restore" and bound.command.request.operation == "restore"
    ):
        os._exit(42)
    return result


WorkerLoop(scope, bind, execute, budget).serve(sys.stdin.buffer, sys.stdout.buffer)
