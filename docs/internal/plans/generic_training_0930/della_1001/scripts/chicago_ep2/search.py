"""Search model capacities with one model-owned resource set live at a time.

Each worker uses the ordinary Trainer search over recomputation and orderings.
Separate processes release CUDA/NCCL/MoonEP resources between capacities. JSON
progress is saved after each candidate; rerunning resumes completed plans.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def launch(config_path, root, world, *, planning):
    command = [
        sys.executable,
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world}",
        "--tee",
        "3",
        "--log-dir",
        str(root / "processes"),
        str(Path(__file__).with_name("train.py")),
        "--config",
        str(config_path),
    ]
    if planning:
        command.append("--plan-only")
    env = dict(os.environ, WANDB_DIR=str(root))
    with (root / ("planning-console.log" if planning else "training-console.log")).open(
        "a"
    ) as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path(__file__).with_name("config.json")
    )
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.steps is not None:
        config["steps"] = args.steps
    root = Path(config["outdir"])
    root.mkdir(parents=True, exist_ok=True)
    cases = []
    for tokens in config["microbatch_candidates"]:
        directory = root / "candidates" / f"tokens-{tokens}"
        directory.mkdir(parents=True, exist_ok=True)
        case_config = {**config, "microbatch_tokens": tokens, "outdir": str(directory)}
        config_path = directory / "config.json"
        if config_path.exists() and json.loads(config_path.read_text()) != case_config:
            raise ValueError(f"Saved candidate configuration differs: {config_path}")
        write(config_path, case_config)
        result_path = directory / "result.json"
        previous = json.loads(result_path.read_text()) if result_path.exists() else {}
        if previous.get("status") == "passed":
            print(f"RESUME tokens/rank={tokens}: completed plan", flush=True)
            record = previous
        else:
            record = dict(
                tokens_per_rank=tokens,
                status="running",
                started_utc=datetime.now(UTC).isoformat(),
            )
            write(result_path, record)
            print(
                f"START planning tokens/rank={tokens} {record['started_utc']}",
                flush=True,
            )
            started = time.monotonic()
            code = launch(config_path, directory, config["world_size"], planning=True)
            record.update(
                exit_code=code,
                elapsed_seconds=time.monotonic() - started,
                finished_utc=datetime.now(UTC).isoformat(),
                status="failed" if code else "passed",
            )
            if not code:
                ranks = [
                    json.loads(
                        (directory / f"rank-{rank:05d}" / "planning.json").read_text()
                    )
                    for rank in range(config["world_size"])
                ]
                record["ranks"] = ranks
                record["predicted_seconds"] = max(
                    rank["predicted_seconds"] for rank in ranks
                )
            write(result_path, record)
            print(f"STOP planning {json.dumps(record)}", flush=True)
        cases.append(record)
        write(root / "planning-progress.json", cases)
    feasible = [case for case in cases if case["status"] == "passed"]
    if not feasible:
        raise RuntimeError(
            "No microbatch candidate produced an admitted plan; inspect candidate logs"
        )
    winner = min(feasible, key=lambda case: case["predicted_seconds"])
    write(root / "selected.json", winner)
    tokens = winner["tokens_per_rank"]
    directory = root / "candidates" / f"tokens-{tokens}"
    print(
        f"SELECTED tokens/rank={tokens}, "
        f"predicted step={winner['predicted_seconds']:.3f}s",
        flush=True,
    )
    code = launch(
        directory / "config.json", directory, config["world_size"], planning=False
    )
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    main()
