"""Capture five EP2 training steps using the admitted 128K model configuration."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = Path(
    "/home/as1669/storage/shadowspill/generic_training_0930/della_1001/chicago-ep2-capacity-fixes-1002"
)


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=BASE / "training-128k-v2/training-config.json"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=BASE / "training-128k-v2/checkpoints/step_00000900",
    )
    parser.add_argument("--outdir", type=Path, default=BASE / "nsys-128k-5steps-1002")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--nsys", default="/usr/local/cuda-13.1/bin/nsys")
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("TMUX"):
        parser.error("Run in the allocated codex tmux pane")
    if min(args.steps, args.warmup) < 1:
        parser.error("Steps and warmup must be positive")
    out = args.outdir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "trace.nsys-rep").exists() or any(out.glob("rank-*/profile-result.json")):
        parser.error("Choose a fresh output directory; an earlier profile exists")
    cfg = json.loads(args.config.read_text())
    cfg.update(outdir=str(out), steps=args.steps + args.warmup)
    cfg.pop("training_end_utc", None)
    write(out / "config.json", cfg)
    repository = next(p for p in HERE.parents if (p / "workloads").is_dir())
    env = dict(os.environ)
    for key in (
        "PYTHONPATH",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "SHADOWSPILL_LIBRARY_DIRECTORY",
    ):
        env.pop(key, None)
    cache = BASE.with_name("chicago-olmoe12b-ep2-1002")
    env.update(
        OMP_NUM_THREADS="8",
        TORCHINDUCTOR_COMPILE_THREADS="1",
        PYTHONUNBUFFERED="1",
        NCCL_SOCKET_IFNAME="lo",
        GLOO_SOCKET_IFNAME="lo",
        WANDB_MODE="disabled",
        TORCHINDUCTOR_CACHE_DIR=str(cache / "inductor_cache"),
        TRITON_CACHE_DIR=str(cache / "triton_cache"),
        CUDA_CACHE_PATH=str(cache / "cuda_cache"),
        TMPDIR=f"/tmp/ss_ep2_nsys_{env['SLURM_JOB_ID']}",
    )
    Path(env["TMPDIR"]).mkdir(exist_ok=True)
    source = out / "source"
    source.mkdir(exist_ok=True)
    hashes = {}
    for name in (
        Path(__file__),
        HERE / "chicago_ep2/train.py",
        HERE / "chicago_ep2/nsys_profile.py",
    ):
        shutil.copyfile(name, source / name.name)
        hashes[name.name] = hashlib.sha256(name.read_bytes()).hexdigest()
    write(source / "sha256.json", hashes)
    command = [
        args.nsys,
        "profile",
        "--trace=cuda,nvtx,osrt,cublas",
        "--sample=none",
        "--cpuctxsw=none",
        "--gpu-metrics-devices=all",
        "--gpu-metrics-frequency=10000",
        "--capture-range=cudaProfilerApi",
        "--capture-range-end=stop",
        "--wait=all",
        "--cuda-graph-trace=node",
        "--export=sqlite",
        "--output=" + str(out / "trace"),
        sys.executable,
        "-u",
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={cfg['world_size']}",
        "--tee",
        "3",
        "--log-dir",
        str(out / "processes"),
        str(HERE / "chicago_ep2/train.py"),
        "--config",
        str(out / "config.json"),
        "--profile-steps",
        str(args.steps),
        "--profile-warmup",
        str(args.warmup),
        "--profile-checkpoint",
        str(args.checkpoint.resolve()),
    ]
    record = dict(
        status="RUNNING",
        started_utc=datetime.now(UTC).isoformat(),
        job=os.environ["SLURM_JOB_ID"],
        command=command,
        capture_steps=args.steps,
        warmup_steps=args.warmup,
    )
    write(out / "status.json", record)
    print(json.dumps(record), flush=True)
    try:
        with (out / "console.log").open("w") as log, subprocess.Popen(
            command,
            cwd=repository,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        if code:
            raise RuntimeError(
                f"Nsight/training exited with code {code}; inspect console.log"
            )
        ranks = [
            json.loads((out / f"rank-{rank:05d}/profile-result.json").read_text())
            for rank in range(cfg["world_size"])
        ]
        if not all(r["passed"] and r["captured_steps"] == args.steps for r in ranks):
            raise RuntimeError("Not every rank captured the requested training steps")
        if (
            not (out / "trace.nsys-rep").is_file()
            or not (out / "trace.sqlite").is_file()
        ):
            raise RuntimeError("Nsight report or SQLite export is missing")
        record.update(status="PASS", ranks=ranks)
    except BaseException as error:
        record.update(status="FAIL", error=repr(error))
        raise
    finally:
        record["finished_utc"] = datetime.now(UTC).isoformat()
        write(out / "status.json", record)
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
