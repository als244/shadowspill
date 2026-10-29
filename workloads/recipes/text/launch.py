"""Workers on several machines: commands, remote invocation, copies.

The first machine is the one the runner is launched on; every other is
reached by ssh (``{"address", "ranks", "ssh", "repo", "python", "env"}``),
with the repository and its environment installed and the config's paths
resolving there. Grouped phases run under one rendezvous at the first
machine's address, each node advertising its own address, so no name
resolution between the machines is needed.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys


def check_nodes(nodes):
    """The nodes entry, checked: at least two, the first local, every other
    with ssh, repo and python, each with a positive rank count."""
    if not isinstance(nodes, list) or len(nodes) < 2:
        raise ValueError(
            "nodes lists at least two machines; one machine needs only ranks"
        )
    for index, node in enumerate(nodes):
        if not isinstance(node, dict) or "address" not in node:
            raise ValueError("every node has an address")
        if type(node.get("ranks", 1)) is not int or node.get("ranks", 1) < 1:
            raise ValueError("a node's ranks is a positive integer")
        if index == 0 and node.get("ssh"):
            raise ValueError("the first node is the one the runner is launched on")
        if index and not all(node.get(key) for key in ("ssh", "repo", "python")):
            raise ValueError("a remote node names ssh, repo and python")
    return nodes


def node_ranks(nodes):
    return [int(node.get("ranks", 1)) for node in nodes]


def first_rank(nodes, index):
    return sum(node_ranks(nodes)[:index])


def worker_command(
    python,
    module,
    root,
    *,
    nodes=1,
    node=0,
    ranks=1,
    endpoint=None,
    address=None,
    identity="run",
):
    """One node's command for a phase: a plain interpreter for a
    single-process phase, otherwise torchrun under the group's rendezvous."""
    if nodes == 1 and ranks == 1:
        return [python, "-u", "-m", module, str(root)]
    command = [
        python,
        "-m",
        "torch.distributed.run",
        f"--nnodes={nodes}",
        f"--node-rank={node}",
        f"--nproc-per-node={ranks}",
        "--rdzv-backend=c10d",
        f"--rdzv-endpoint={endpoint}",
        f"--rdzv-id={identity}",
    ]
    if address:
        command.append(f"--local-addr={address}")
    return [
        *command,
        "--no-python",
        python,
        "-u",
        "-m",
        "benchmarking.device_worker",
        module,
        str(root),
    ]


def remote_command(node, command, environment):
    """The ssh invocation running ``command`` in the node's repository."""
    assignments = " ".join(
        f"{key}={shlex.quote(str(value))}" for key, value in environment.items()
    )
    inner = f"cd {shlex.quote(node['repo'])} && env {assignments} {shlex.join(command)}"
    return ["ssh", "-o", "BatchMode=yes", node["ssh"], inner]


def remote(node, command):
    subprocess.run(["ssh", "-o", "BatchMode=yes", node["ssh"], command], check=True)


def copy_to(node, paths, root):
    """Copy ``paths`` (relative to ``root``) to the node at the same place."""
    existing = [str(root / path) for path in paths if (root / path).exists()]
    if existing:
        remote(node, f"mkdir -p {shlex.quote(str(root))}")
        subprocess.run(["rsync", "-a", *existing, f"{node['ssh']}:{root}/"], check=True)


def copy_back(node, paths, root):
    """Copy ``paths`` (relative to ``root``) from the node to the same place here."""
    for path in paths:
        subprocess.run(
            ["rsync", "-a", f"{node['ssh']}:{root / path}/", f"{root / path}/"],
            check=False,
        )


def run_phase(nodes, module, roots, logs, phase, *, grouped, port, environment=None):
    """Run ``module`` for every node at once, each in its own ``roots[i]``:
    one process per node for a plain phase, the group's workers for a grouped
    one. Raises with the failing nodes' log tails."""
    endpoint = f"{nodes[0]['address']}:{port}"
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    for index, node in enumerate(nodes):
        settings = {**node.get("env", {}), **(environment or {})}
        if grouped:
            settings.setdefault("OMP_NUM_THREADS", "1")
        python = node.get("python") if index else sys.executable
        command = worker_command(
            python or sys.executable,
            module,
            roots[index],
            nodes=len(nodes) if grouped else 1,
            node=index,
            ranks=node.get("ranks", 1) if grouped else 1,
            endpoint=endpoint,
            address=node["address"],
            identity=f"{roots[0].name}-{phase}",
        )
        log = logs / (f"{phase}.log" if index == 0 else f"{phase}-node{index}.log")
        stream = log.open("w")
        if index == 0:
            process = subprocess.Popen(
                command,
                env={**os.environ, **settings},
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        else:
            process = subprocess.Popen(
                remote_command(node, command, settings),
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        processes.append((node, log, stream, process))
    failures = []
    for node, log, stream, process in processes:
        status = process.wait()
        stream.close()
        if status:
            tail = "".join(log.read_text().splitlines(keepends=True)[-20:])
            failures.append(
                f"{node['address']} status {status}; the last lines of {log}:\n{tail}"
            )
    if failures:
        raise RuntimeError(
            f"{phase} failed on {len(failures)} node(s):\n" + "\n".join(failures)
        )


def rank_directories(nodes, index):
    first = first_rank(nodes, index)
    return [
        f"rank-{rank:05d}" for rank in range(first, first + node_ranks(nodes)[index])
    ]
