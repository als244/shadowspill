"""Expose only selected healthy device files for symmetric-planning DP checks."""

import argparse
import resource
import shlex
import socket
import subprocess
from datetime import UTC, datetime
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "mode",
    choices=(
        "health",
        "auto",
        "auto-control",
        "recompute",
        "sweep",
        "sweep-whole",
        "capture-control",
        "llama",
        "bias",
        "alias",
        "compiler",
        "suite",
    ),
)
parser.add_argument("--world-size", type=int, choices=(2, 4), default=2)
parser.add_argument("--no-symmetric-planning", action="store_true")
parser.add_argument("--probe-args", nargs=argparse.REMAINDER, default=[])
args = parser.parse_args()
root = Path.home() / "shadowspill"
plan = root / "docs/internal/plans/qwen_moe_ep8_1007"
env = Path.home() / "miniconda3/envs/shadowspill"
uuids = [
    "GPU-14013517-e066-7ce0-1d72-fb10083cd905",
    "GPU-af041483-6cec-9cb4-0453-5d213e479f2e",
    "GPU-2a71ac7c-28d8-566b-bef7-49f5b71230a6",
    "GPU-d57829eb-02db-bf9f-0934-882f7feecdb3",
][: args.world_size]
validation_root = plan / "evidence/fatnode"
if args.world_size != 2:
    validation_root /= f"dp{args.world_size}"
container_name = f"shadowspill-symmetric-dp{args.world_size}"
minors = {}
for path in Path("/proc/driver/nvidia/gpus").glob("*/information"):
    fields = dict(
        line.split(":", 1) for line in path.read_text().splitlines() if ":" in line
    )
    minors[fields["GPU UUID"].strip()] = int(fields["Device Minor"])
selected = [minors[uuid] for uuid in uuids]
assert len(set(selected)) == args.world_size
hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)[1]
limit = -1 if hard == resource.RLIM_INFINITY else hard
command = [
    "podman",
    "run",
    "--rm",
    "--name",
    container_name,
    "--network=host",
    "--shm-size=1g",
    "--security-opt=label=disable",
    "--ulimit",
    f"memlock={limit}:{limit}",
    "--workdir",
    str(root),
]
for minor in selected:
    command += ["--device", f"/dev/nvidia{minor}"]
for device in ("/dev/nvidiactl", "/dev/nvidia-uvm"):
    command += ["--device", device]
for path in (
    "/usr/bin/git",
    "/usr/lib/git-core",
    "/usr/share/git-core",
    "/usr/bin/nvidia-smi",
):
    command += ["--volume", f"{path}:{path}:ro"]
for name in (
    "libnvidia-ml.so.1",
    "libcuda.so.1",
    "libnvidia-ptxjitcompiler.so.1",
    "libnvidia-nvvm.so.4",
):
    path = Path("/usr/lib/x86_64-linux-gnu") / name
    if path.exists():
        command += ["--volume", f"{path.resolve()}:{path}:ro"]
for path, mode in (
    (env, "ro"),
    (root, "rw"),
    (Path.home() / "mlops", "ro"),
    (Path("/usr/local/cuda-13.4"), "ro"),
):
    command += ["--volume", f"{path}:{path}:{mode}"]
for name, value in {
    "CUDA_VISIBLE_DEVICES": ",".join(uuids),
    "CUDA_HOME": "/usr/local/cuda-13.4",
    "PYTHONPATH": str(root),
    "PATH": f"{env}/bin:/usr/local/cuda-13.4/bin:/usr/local/bin:/usr/bin:/bin",
    "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu",
    "NCCL_DEBUG": "WARN",
    "NCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "TORCHINDUCTOR_COMPILE_THREADS": "1",
    "TORCHINDUCTOR_CACHE_DIR": str(plan / "evidence/fatnode/cache/inductor"),
    "TRITON_CACHE_DIR": str(plan / "evidence/fatnode/cache/triton"),
}.items():
    command += ["--env", f"{name}={value}"]
with socket.socket() as listener:
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
command += [
    "localhost/shadowspill-validation:ubuntu24.04",
    str(env / "bin/python"),
    "-u",
    "-m",
]
if args.mode == "suite":
    command += ["qualification.gates", "suite", "--run", "bias_functionalization_1008"]
elif args.mode == "compiler":
    command += [
        "pytest",
        "-o",
        "addopts=",
        "-q",
        "tests/shadowspill/pytorch/compilation",
    ]
else:
    command += [
        "torch.distributed.run",
        "--nnodes=1",
        f"--nproc-per-node={args.world_size}",
        "--master-addr=127.0.0.1",
        f"--master-port={port}",
    ]
if args.mode in ("suite", "compiler"):
    pass
elif args.mode in ("bias", "alias"):
    output = validation_root / (
        args.mode + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    )
    command += [
        str(
            plan
            / "scripts/fatnode"
            / ("bias_probe.py" if args.mode == "bias" else "alias_repro.py")
        ),
        "--out",
        str(output),
        *args.probe_args,
    ]
elif args.mode == "llama":
    label = "llama-symmetric" if not args.no_symmetric_planning else "llama-independent"
    output = validation_root / (
        label + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    )
    source = (
        root
        / "docs/internal/plans/fatnode_refresh_1006/preserved_0930/shadowspill"
        / "docs/internal/plans/generic_training_0930/experiments/fineweb_dp_1001"
    )
    command += [
        str(plan / "scripts/fatnode/llama.py"),
        "--inputs-root",
        str(source),
        "--outdir",
        str(output),
    ]
    if args.no_symmetric_planning:
        command += ["--no-symmetric-planning"]
elif args.mode in ("health", "sweep", "sweep-whole", "capture-control"):
    command += [
        str(
            plan
            / "scripts/fatnode"
            / ("health.py" if args.mode == "health" else "search.py")
        )
    ]
    if args.mode == "sweep-whole":
        command += ["--partition", "whole"]
    elif args.mode == "capture-control":
        command += ["--mutable-buffer", "--no-symmetric-planning"]
    if args.mode != "health":
        label = {
            "sweep": "sweep-auto",
            "sweep-whole": "sweep-whole",
            "capture-control": "capture-control",
        }[args.mode]
        command += ["--outdir", str(validation_root / label)]
        if args.world_size == 4:
            command += ["--rows-per-rank", "8"]
else:
    output = validation_root / args.mode
    if output.exists():
        output = output.with_name(
            args.mode + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
        )
    command += [
        "-m",
        "tests.shadowspill.pytorch.distributed._training_case",
        "--out",
        str(output),
        "--variant",
        "auto" if args.mode == "auto-control" else args.mode,
    ]
    if args.mode != "auto-control":
        command += ["--symmetric-planning"]
    if args.mode == "recompute":
        command += [
            "--precision",
            "fp16",
            "--masters",
            "--optimizer",
            "mlops",
            "--diagnostics",
        ]
    else:
        command += ["--parameter-metrics"]
print("Healthy GPU UUIDs:", uuids, "device minors:", selected, flush=True)
print(shlex.join(command), flush=True)
try:
    status = subprocess.run(
        command,
        timeout=180
        if args.mode == "health"
        else (5400 if args.mode in ("llama", "suite") else 900),
    ).returncode
except subprocess.TimeoutExpired:
    subprocess.run(
        ["podman", "stop", "--time", "10", container_name],
        timeout=30,
        check=False,
    )
    raise
print(f"CONTAINER_EXIT mode={args.mode} status={status}", flush=True)
raise SystemExit(status)
