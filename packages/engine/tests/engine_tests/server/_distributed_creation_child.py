"""Real CPU Gloo startup/retirement through the service-owned worker launcher."""

import os
import socket
import sys
from types import SimpleNamespace

from shadowspill.orchestration.launch.workers import workers
from shadowspill.training.worker.channel import read_message
from shadowspill.training.worker.loop import WorkerLoop
from shadowspill.training.worker.startup import await_start, report_ready
from shadowspill_client.buffers import BufferBudget

with (
    socket.socket(fileno=int(sys.argv[1])) as peer,
    peer.makefile("rwb", buffering=0) as stream,
):
    launch = read_message(stream)
    await_start(launch.startup, stream, stream)
    mapping = dict(enumerate(launch.startup.logical_by_worker))
    with workers(
        len(mapping),
        rank_map=mapping,
        init_method=os.environ["SHADOWSPILL_RENDEZVOUS"],
    ) as group:
        assert group.logical_rank == launch.startup.scope.rank
        assert group.gather(group.logical_rank) == list(mapping.values())
        report_ready(launch.startup, stream)

        def bind(command):
            return SimpleNamespace(
                command=command,
                checkpoint=None,
                restore=None,
                verify_inputs=lambda _: None,
            )

        WorkerLoop(
            launch.startup.scope, bind, lambda _: None, BufferBudget(1)
        ).serve(stream, stream)
print("service-owned distributed CPU worker exited")
