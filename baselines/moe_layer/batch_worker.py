"""Run a sequence of cases with shared imports, CUDA state and process group."""

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from baselines.common import write_json

from .session import ModelSession
from .worker import main as run_case


@contextmanager
def case_log(path):
    # Redirect native CUDA/NCCL output too. O_APPEND allows all ranks to share
    # the same per-case console log, as the isolated torchrun launcher does.
    sys.stdout.flush()
    sys.stderr.flush()
    saved = [os.dup(fd) for fd in (1, 2)]
    try:
        with path.open("ab", buffering=0) as stream:
            os.dup2(stream.fileno(), 1)
            os.dup2(stream.fileno(), 2)
            yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        for fd, original in zip((1, 2), saved, strict=True):
            os.dup2(original, fd)
            os.close(original)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    args = parser.parse_args()
    queue = json.loads(args.queue.read_text())
    rank = int(os.environ.get("RANK", "0"))
    cases = queue["completed_cases"]
    session = ModelSession()
    for job in queue["jobs"]:
        attempt = Path(job["attempt"])
        attempt.mkdir(parents=True, exist_ok=True)
        started = time.time()
        if rank == 0:
            write_json(attempt / "command.json", job["command"])
            write_json(
                args.queue.parent / "current.json",
                {
                    "case": job["case"],
                    "attempt": str(attempt),
                    "started": started,
                },
            )
            print(f"Starting {job['case']} (shared worker session)", flush=True)
        with case_log(attempt / "console.log"):
            # On failure, fail the complete torchrun group. The controller can
            # start the remaining cases in fresh workers after a genuine OOM.
            run_case(job["arguments"], session=session)
        import torch.distributed as dist

        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            result = json.loads((attempt / "result.json").read_text())
            if result.get("all_ranks_validated") != queue["ranks"]:
                raise RuntimeError("Incomplete rank validation")
            outcome = {
                "status": "completed",
                "returncode": 0,
                "attempt": str(attempt.relative_to(queue["root"])),
                "elapsed_seconds": time.time() - started,
                "shared_worker_session": str(args.queue.parent),
            }
            write_json(attempt / "outcome.json", outcome)
            cases.append({"case": job["case"], **outcome})
            write_json(Path(queue["root"]) / "progress.json", {"cases": cases})
            print(f"Finished {job['case']}: completed", flush=True)
    session.close()
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()
    if rank == 0:
        write_json(args.queue.parent / "finished.json", {"status": "completed"})


if __name__ == "__main__":
    main()
