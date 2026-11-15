"""Owned CPU process exercising the actual checkpoint command/stream encoder."""

import hashlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine_tests.execution.test_checkpoint import MemoryExecutor
from shadowspill.training.session.controls import CallControls
from shadowspill.training.worker.bindings import BoundWorkerCall
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
scope = WorkerScope("run", "server", f"process-{rank}", rank)
budget = BufferBudget(64 << 10)
writer = WorkerCheckpointWriter(
    executor,
    budget,
    sys.stdout.buffer,
    plans=ArtifactRef("plans", hashlib.sha256(b"plans").hexdigest(), 5),
    chunk_bytes=32,
)
execution = WorkerExecution(
    executor,
    SimpleNamespace(scope=scope),
    budget,
    lambda *args: None,
    checkpoints=writer,
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
    if mode == "lost_completion":
        os._exit(42)
    return result


WorkerLoop(scope, bind, execute, budget).serve(sys.stdin.buffer, sys.stdout.buffer)
