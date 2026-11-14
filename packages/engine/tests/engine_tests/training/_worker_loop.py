"""CPU process-fault fixture; no model/execution correctness claims."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

from shadowspill.training.worker.loop import WorkerLoop
from shadowspill.training.worker.protocol import WorkerScope
from shadowspill_client.buffers import BufferBudget

marker, mode = Path(sys.argv[1]), sys.argv[2]


def bind(command):
    return SimpleNamespace(command=command, verify_inputs=lambda budget: None)


def execute(bound):
    with marker.open("a") as output:
        output.write(f"{bound.command.invocation}\n")
        output.flush()
        os.fsync(output.fileno())
    if mode == "lost_completion":
        os._exit(42)


WorkerLoop(
    WorkerScope("run", "server", "process-1", 0), bind, execute, BufferBudget(1)
).serve(sys.stdin.buffer, sys.stdout.buffer)
