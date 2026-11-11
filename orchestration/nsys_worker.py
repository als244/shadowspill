"""One Nsight collector per global rank, launched before importing CUDA.

Use NsysCapture.rank_command inside torchrun (including multi-node torchrun).
The collector inherits exactly one CUDA-visible GPU, so hardware metrics are
also restricted to that GPU. No torch/distributed import or rendezvous happens
here: the child training worker owns those operations.
"""

import argparse
import json
import os
import socket
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

from .nsys import NsysCapture
from .rank_device import isolate_device


def collect(command, *, capture, output_dir):
    """Run one worker and require a nonempty report before returning success."""
    environment = dict(os.environ)
    rank = int(environment.get("RANK", "0"))
    local_rank = int(environment.get("LOCAL_RANK", "0"))
    world_size = int(environment.get("WORLD_SIZE", "1"))
    if not 0 <= rank < world_size:
        raise ValueError("RANK must be within WORLD_SIZE")
    original_mask = environment.get("CUDA_VISIBLE_DEVICES")
    device = isolate_device(environment)
    directory = Path(output_dir).expanduser().resolve() / f"rank-{rank:05d}"
    report = directory / "trace.nsys-rep"
    collector = capture.command(command, output=report.with_suffix(""))
    directory.mkdir(parents=True, exist_ok=False)
    metadata = dict(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        hostname=socket.gethostname(),
        original_cuda_visible_devices=original_mask,
        cuda_visible_devices=device,
        settings=asdict(capture),
        command=collector,
        report_path=report.name,
        content_verified=False,
        status="running",
        started_unix_ns=time.time_ns(),
    )

    def record():
        path = directory / "capture.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(path)

    record()
    try:
        # Keep the CUDA mask on the collector, not merely its child worker.
        # This scopes --gpu-metrics-devices=cuda-visible to this rank's GPU.
        with (directory / "collector.log").open("x") as log:
            result = subprocess.run(
                collector, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
        metadata["returncode"] = result.returncode
        if result.returncode:
            raise RuntimeError(
                f"rank {rank} Nsight/worker failed ({result.returncode}); "
                f"see {directory / 'collector.log'}"
            )
        if not report.is_file() or not report.stat().st_size:
            raise RuntimeError(f"Nsight exited without a nonempty report: {report}")
        metadata.update(status="complete", report_bytes=report.stat().st_size)
    except BaseException as error:
        metadata.update(status="failed", error=repr(error))
        raise
    finally:
        metadata["finished_unix_ns"] = time.time_ns()
        record()
    return metadata


def reports(output_dir, *, ranks):
    """Validate completed collectors after the distributed launcher exits."""
    root = Path(output_dir)
    found = []
    for rank in range(ranks):
        directory = root / f"rank-{rank:05d}"
        path = directory / "capture.json"
        if not path.is_file():
            raise RuntimeError(f"missing Nsight rank {rank} capture manifest: {path}")
        metadata = json.loads(path.read_text())
        if (
            metadata.get("status") != "complete"
            or metadata.get("rank") != rank
            or metadata.get("world_size") != ranks
        ):
            raise RuntimeError(f"incomplete or mismatched Nsight capture: {path}")
        report = directory / "trace.nsys-rep"
        if (
            not report.is_file()
            or not report.stat().st_size
            or report.stat().st_size != metadata.get("report_bytes")
        ):
            raise RuntimeError(f"missing or changed Nsight report: {report}")
        found.append(dict(metadata, report_path=str(report.relative_to(root))))
    return found


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="{}", help="NsysCapture fields as JSON")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args(argv)
    command = options.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a worker command is required after --")
    collect(
        command,
        capture=NsysCapture(**json.loads(options.config)),
        output_dir=options.output_dir,
    )


if __name__ == "__main__":
    main()
