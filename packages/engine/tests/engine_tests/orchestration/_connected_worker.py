"""Private inherited socket and process faults; no GPU/model claims."""

import os
import socket
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from shadowspill.training.worker.loop import WorkerLoop
from shadowspill.training.worker.protocol import WorkerScope, WorkerStartup
from shadowspill.training.worker.startup import await_start, report_ready
from shadowspill_client.buffers import BufferBudget
from shadowspill_client.protocol.requests import ArtifactRef

fd, mode, marker = int(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
expected = WorkerStartup(
    WorkerScope("run", "server", "process-1", 0),
    ArtifactRef("plans", "a" * 64, 128),
    "b" * 64,
    (0,),
    2000,
    1000,
    500,
)


def record(value):
    with marker.open("a") as output:
        output.write(value + "\n")
        output.flush()
        os.fsync(output.fileno())


def bind(command):
    return SimpleNamespace(command=command, verify_inputs=lambda _: None)


def execute(bound):
    record(str(bound.command.invocation))
    if mode == "lost_completion":
        os._exit(42)


with (
    socket.socket(fileno=fd) as channel,
    channel.makefile("rwb", buffering=0) as stream,
):
    if mode == "stall":
        time.sleep(60)
    for output in (1, 2):
        os.write(output, b"x" * (1 << 20))
    await_start(expected, stream, stream)
    record("initialized")
    if mode == "lost_ready":
        os._exit(42)
    report_ready(expected, stream)
    WorkerLoop(expected.scope, bind, execute, BufferBudget(1)).serve(stream, stream)
    record("closed")
