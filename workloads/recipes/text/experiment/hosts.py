"""Several hosts, one run: each builds for its own GPUs, all train together.

A config's ``hosts`` lists the machines, the first being the one the runner
is launched on: ``{"address": ..., "ranks": N, "ssh": alias, "repo": path,
"python": path, "env": {...}}``. The build, profile and plan phases run on
every host at once, each under its own target directory; the communicate and
execute phases launch every host's workers under one rendezvous at the first
host's address, so the ranks form one group over the network. Remote hosts
need the repository and its environment, and the config's paths (run
directory, data, checkpoint) must resolve on each of them. Remote ranks'
directories are copied back to the first host after execution.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys

PACKAGE = "workloads.recipes.text.experiment"
PHASES = ("build", "profile", "communicate", "plan", "execute")


def check_hosts(raw):
    """The hosts entry, checked: the first is local, every other reachable by
    ssh with a repository and an interpreter, and ranks add up."""
    hosts = raw.get("hosts")
    if hosts is None:
        return None
    if not isinstance(hosts, list) or len(hosts) < 2:
        raise ValueError(
            "hosts lists at least two machines; one machine needs only ranks"
        )
    for index, host in enumerate(hosts):
        if not isinstance(host, dict) or "address" not in host:
            raise ValueError("every host has an address")
        if type(host.get("ranks", 1)) is not int or host.get("ranks", 1) < 1:
            raise ValueError("a host's ranks is a positive integer")
        if index == 0 and host.get("ssh"):
            raise ValueError("the first host is the one the runner is launched on")
        if index and not all(host.get(key) for key in ("ssh", "repo", "python")):
            raise ValueError("a remote host names ssh, repo and python")
    total = sum(host.get("ranks", 1) for host in hosts)
    if raw.get("ranks", total) != total:
        raise ValueError("ranks must equal the hosts' ranks added up")
    if raw.get("max_tokens_per_microbatch") is None:
        raise ValueError(
            "hosts need a fixed max_tokens_per_microbatch, the one microbatch "
            "capacity every target packs"
        )
    return hosts


def worker_command(
    python, module, root, *, nodes=1, node=0, ranks=1, endpoint=None, identity="run"
):
    """One host's command for a phase: a plain interpreter for a single-process
    phase, otherwise torchrun under the group's rendezvous."""
    if nodes == 1 and ranks == 1:
        return [python, "-u", "-m", module, str(root)]
    return [
        python,
        "-m",
        "torch.distributed.run",
        f"--nnodes={nodes}",
        f"--node-rank={node}",
        f"--nproc-per-node={ranks}",
        "--rdzv-backend=c10d",
        f"--rdzv-endpoint={endpoint}",
        f"--rdzv-id={identity}",
        "--no-python",
        python,
        "-u",
        "-m",
        "benchmarking.device_worker",
        module,
        str(root),
    ]


def remote_command(host, command, environment):
    """The ssh invocation running ``command`` in the host's repository."""
    assignments = " ".join(
        f"{key}={shlex.quote(str(value))}" for key, value in environment.items()
    )
    inner = f"cd {shlex.quote(host['repo'])} && env {assignments} {shlex.join(command)}"
    return ["ssh", "-o", "BatchMode=yes", host["ssh"], inner]


def ssh(host, command):
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", host["ssh"], command],
        check=True,
        capture_output=True,
        text=True,
    )


def run_hosts(root, raw, selected, hosts):
    """Run the selected phase, or all, across the hosts."""
    port = raw.get("port", 29400)
    endpoint = f"{hosts[0]['address']}:{port}"
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for host in hosts[1:]:
        ssh(host, f"mkdir -p {shlex.quote(str(root / 'logs'))}")
        subprocess.run(
            [
                "scp",
                "-q",
                str(root / "config.json"),
                f"{host['ssh']}:{root / 'config.json'}",
            ],
            check=True,
        )
    nodes = len(hosts)
    for phase in PHASES:
        if selected not in {"all", phase}:
            continue
        grouped = phase in {"communicate", "execute"}
        print(
            f"{phase}: starting on {nodes} hosts; logs {logs}/{phase}*.log", flush=True
        )
        processes = []
        for index, host in enumerate(hosts):
            environment = {**host.get("env", {}), "SHADOWSPILL_HOST_INDEX": str(index)}
            if phase == "plan":
                environment["CUDA_VISIBLE_DEVICES"] = ""
            if grouped:
                environment.setdefault("OMP_NUM_THREADS", "1")
            python = host.get("python", sys.executable) if index else sys.executable
            command = worker_command(
                python,
                f"{PACKAGE}.{phase}",
                root,
                nodes=nodes if grouped else 1,
                node=index,
                ranks=host.get("ranks", 1) if grouped else 1,
                endpoint=endpoint,
                identity=root.name,
            )
            log = logs / (f"{phase}.log" if index == 0 else f"{phase}-host{index}.log")
            stream = log.open("w")
            if index == 0:
                process = subprocess.Popen(
                    command,
                    env={**os.environ, **environment},
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            else:
                process = subprocess.Popen(
                    remote_command(host, command, environment),
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
            processes.append((host, log, stream, process))
        failures = []
        for host, log, stream, process in processes:
            status = process.wait()
            stream.close()
            if status:
                tail = "".join(log.read_text().splitlines(keepends=True)[-20:])
                failures.append(
                    f"{host['address']} status {status}; "
                    f"the last lines of {log}:\n{tail}"
                )
        if failures:
            raise RuntimeError(
                f"{phase} failed on {len(failures)} host(s):\n" + "\n".join(failures)
            )
        print(f"{phase}: complete", flush=True)
    if selected in {"all", "execute"}:
        for index, host in enumerate(hosts[1:], start=1):
            first = sum(h.get("ranks", 1) for h in hosts[:index])
            for rank in range(first, first + host.get("ranks", 1)):
                name = f"rank-{rank:05d}"
                subprocess.run(
                    [
                        "rsync",
                        "-a",
                        f"{host['ssh']}:{root / name}/",
                        str(root / name) + "/",
                    ],
                    check=False,
                )
