"""CPU private-channel streaming fixture, with intentional owned-process loss."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

from shadowspill.training.worker.binary import WorkerStream
from shadowspill.training.worker.loop import WorkerLoop
from shadowspill.training.worker.protocol import WorkerScope
from shadowspill_client.buffers import BufferBudget

marker, mode = Path(sys.argv[1]), sys.argv[2]
rank = int(sys.argv[3]) if len(sys.argv) > 3 else 0
SIZE = (5 << 16) + 17


def bind(command):
    return SimpleNamespace(
        command=command,
        checkpoint=None,
        restore=None,
        verify_inputs=lambda budget: None,
    )


def execute(bound):
    with marker.open("a") as output:
        output.write(f"{bound.command.invocation}\n")
        output.flush()
        os.fsync(output.fileno())
    with WorkerStream(sys.stdout.buffer, size_bytes=SIZE) as output:
        for index in range(5):
            output.write(b"x" * (1 << 16))
            if mode == "lost_stream" and index == 1:
                sys.stdout.buffer.flush()
                os._exit(42)
        output.write(b"x" * 17)
    if mode == "lost_completion":
        os._exit(42)
    scope = f"{rank}-" if len(sys.argv) > 3 else ""
    ref = output.reference(
        f"output-{scope}{bound.command.invocation}", "application/octet-stream"
    )
    if mode == "bad_reference":
        from dataclasses import replace

        ref = replace(ref, size_bytes=SIZE + 1)
    return ref


WorkerLoop(
    WorkerScope("run", "server", "process-1", rank), bind, execute, BufferBudget(1)
).serve(sys.stdin.buffer, sys.stdout.buffer)
