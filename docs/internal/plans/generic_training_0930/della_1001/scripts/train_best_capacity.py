"""Train the fastest admitted EP2 capacity after the complete planning sweep.

Runs in the allocated codex pane. Reuses the winning case's artifact store and
keeps the configured LR horizon independent of the requested training length.
"""

import argparse
import json
import math
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

from chicago_ep2.search import launch, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path(
            "/home/as1669/storage/shadowspill/generic_training_0930/della_1001/chicago-ep2-capacity-fixes-1002"
        ),
    )
    length = parser.add_mutually_exclusive_group()
    length.add_argument("--steps", type=int)
    length.add_argument(
        "--until-allocation-end",
        action="store_true",
        help="Size training from the startup trace, reserving ten minutes for shutdown",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Prepared token directory; defaults to the planning sample",
    )
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument(
        "--training-outdir",
        type=Path,
        help="Fresh run directory; reuse the selected case's compilation/profile store",
    )
    args = parser.parse_args()
    args.outdir = args.outdir.resolve()
    cases = []
    for tokens in (32768, 65536, 131072, 262144):
        result = args.outdir / f"tokens-{tokens}/result.json"
        record = json.loads(result.read_text()) if result.exists() else {}
        if record.get("status") not in ("passed", "failed"):
            raise RuntimeError(f"Planning has not finished for {tokens} tokens/rank")
        if record["status"] == "passed":
            predicted = max(rank["predicted_seconds"] for rank in record["ranks"])
            if not math.isfinite(predicted) or predicted <= 0:
                raise ValueError(f"Invalid prediction in {result}")
            cases.append({**record, "predicted_seconds": predicted})
    if not cases:
        raise RuntimeError("No capacity passed planning and physical admission")
    winner = min(cases, key=lambda case: case["predicted_seconds"])
    write(args.outdir / "selected-training.json", winner)
    directory = args.outdir / f"tokens-{winner['tokens_per_rank']}"
    config = json.loads((directory / "config.json").read_text())
    if args.training_outdir is not None:
        config["artifact_store"] = str(directory)
        directory = args.training_outdir.resolve()
        directory.mkdir(parents=True, exist_ok=True)
        config["outdir"] = str(directory)
    if args.steps is not None:
        if args.steps <= 0:
            raise ValueError("--steps must be positive")
        config["steps"] = args.steps
    if args.data is not None:
        config["data"] = str(args.data.resolve())
    metadata = json.loads((Path(config["data"]) / "meta.json").read_text())
    if args.until_allocation_end:
        job = os.environ.get("SLURM_JOB_ID")
        if not job:
            raise RuntimeError(
                "An allocation is required to determine its remaining time"
            )
        end_text = subprocess.check_output(
            ["squeue", "-h", "-j", job, "-o", "%e"],
            text=True,
            env={**os.environ, "TZ": "UTC"},
        ).strip()
        end = datetime.fromisoformat(end_text).replace(tzinfo=UTC)
        config["training_end_utc"] = (end - timedelta(minutes=10)).isoformat()
        config["steps"] = min(
            config["schedule_total_steps"],
            metadata["train_tokens"] // config["tokens_per_step"],
        )
    if config["steps"] * config["tokens_per_step"] > metadata["train_tokens"]:
        raise ValueError(
            "Requested run exceeds the prepared training prefix; "
            "supply more data with --data"
        )
    config["wandb_group"] = args.outdir.name
    print(
        f"SELECTED tokens/rank={winner['tokens_per_rank']}, predicted step="
        f"{winner['predicted_seconds']:.3f}s, training update cap={config['steps']}, "
        f"LR horizon={config['schedule_total_steps']}",
        flush=True,
    )
    if args.select_only:
        return
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run GPU training inside the codex Slurm allocation")
    if any(directory.glob("rank-*/metrics.jsonl")):
        raise FileExistsError(
            "This case already has training results; refusing a fresh overwrite"
        )
    repository = next(
        p for p in Path(__file__).resolve().parents if (p / "workloads").is_dir()
    )
    os.chdir(repository)
    for name in (
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "SHADOWSPILL_LIBRARY_DIRECTORY",
    ):
        os.environ.pop(name, None)
    cache = args.outdir.with_name("chicago-olmoe12b-ep2-1002")
    os.environ.update(
        OMP_NUM_THREADS="8",
        TORCHINDUCTOR_COMPILE_THREADS="1",
        PYTHONUNBUFFERED="1",
        NCCL_SOCKET_IFNAME="lo",
        GLOO_SOCKET_IFNAME="lo",
        HTTP_PROXY="http://127.0.0.1:18375",
        HTTPS_PROXY="http://127.0.0.1:18375",
        NO_PROXY="localhost,127.0.0.1",
        WANDB_MODE="online",
        WANDB_DIR=str(directory),
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor_cache"),
        TRITON_CACHE_DIR=str(cache / "triton_cache"),
        CUDA_CACHE_PATH=str(cache / "cuda_cache"),
        TMPDIR=f"/tmp/ss_ep2_{os.environ['SLURM_JOB_ID']}",
    )
    Path(os.environ["TMPDIR"]).mkdir(exist_ok=True)
    import wandb

    if wandb.Api(timeout=20).viewer is None:
        raise RuntimeError(
            "Online W&B authentication failed; inspect the head-node relay"
        )
    print("Online W&B authentication passed", flush=True)
    config_path = directory / "training-config.json"
    write(config_path, config)
    code = launch(config_path, directory, config["world_size"], planning=False)
    if code:
        raise SystemExit(code)
    ranks = [
        json.loads((directory / f"rank-{rank:05d}/completed.json").read_text())
        for rank in range(config["world_size"])
    ]
    targets = {rank["requested_steps"] for rank in ranks}
    if len(targets) != 1 or not all(
        rank["passed"] and rank["steps"] == rank["requested_steps"] <= config["steps"]
        for rank in ranks
    ):
        raise RuntimeError("Not all ranks completed the requested training updates")
    write(args.outdir / "training-completed.json", {"selected": winner, "ranks": ranks})


if __name__ == "__main__":
    main()
