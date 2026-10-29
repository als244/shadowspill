"""Orchestrate a heterogeneous run: each node builds for itself, all train as one."""

from __future__ import annotations

import json
import shlex
import subprocess

from workloads.recipes.text.launch import (
    check_nodes,
    copy_back,
    first_rank,
    node_ranks,
    rank_directories,
    remote,
    run_phase,
)

EXPERIMENT = "workloads.recipes.text.experiment"
PACKAGE = "workloads.recipes.text.heterogeneous"
PHASES = ("build", "profile", "communicate", "plan", "execute")


def check_section(raw):
    """The heterogeneous section, checked: nodes, one positive share per
    rank, equal shares within a node (its ranks hold the same hardware), and
    a step that divides into whole units for every share."""
    section = raw["heterogeneous"]
    if not isinstance(section, dict):
        raise ValueError("heterogeneous is a section: nodes, shares, port")
    nodes = check_nodes(section.get("nodes"))
    counts = node_ranks(nodes)
    total = sum(counts)
    shares = section.get("shares", [1] * total)
    if (
        not isinstance(shares, list)
        or len(shares) != total
        or any(type(share) is not int or share < 1 for share in shares)
    ):
        raise ValueError("shares lists one positive integer per rank")
    at = 0
    for count in counts:
        if len(set(shares[at : at + count])) != 1:
            raise ValueError("the ranks of one node take equal shares")
        at += count
    step_units = raw["max_tokens_per_step"] // raw["max_seq_len"]
    if raw["max_tokens_per_step"] % raw["max_seq_len"] or step_units % sum(shares):
        raise ValueError(
            "max_tokens_per_step must divide into whole units for every share"
        )
    return nodes, shares, section.get("port", 29400)


def node_config(raw, nodes, shares, index):
    """The config node ``index`` runs the phases with: the group's size, its
    ranks' tokens per step and its place in the group."""
    step_units = raw["max_tokens_per_step"] // raw["max_seq_len"]
    rank = first_rank(nodes, index)
    units = step_units // sum(shares) * shares[rank]
    return {
        **raw,
        "ranks": sum(node_ranks(nodes)),
        "rank_tokens_per_step": units * raw["max_seq_len"],
        "node": index,
    }


def node_root(root, index):
    return root / "nodes" / f"{index:02d}"


def run_heterogeneous(root, raw, selected):
    nodes, shares, port = check_section(raw)
    roots = [node_root(root, index) for index in range(len(nodes))]
    logs = root / "logs"
    for index, node in enumerate(nodes):
        config = json.dumps(node_config(raw, nodes, shares, index), indent=2) + "\n"
        roots[index].mkdir(parents=True, exist_ok=True)
        (roots[index] / "config.json").write_text(config)
        if index:
            remote(node, f"mkdir -p {shlex.quote(str(roots[index]))}")
            subprocess.run(
                [
                    "scp",
                    "-q",
                    str(roots[index] / "config.json"),
                    f"{node['ssh']}:{roots[index] / 'config.json'}",
                ],
                check=True,
            )
    for phase in PHASES:
        if selected not in {"all", phase}:
            continue
        grouped = phase in {"communicate", "execute"}
        package = PACKAGE if grouped else EXPERIMENT
        print(
            f"{phase}: starting on {len(nodes)} nodes; logs {logs}/{phase}*.log",
            flush=True,
        )
        run_phase(
            nodes,
            f"{package}.{phase}",
            roots,
            logs,
            phase,
            grouped=grouped,
            port=port,
            environment={"CUDA_VISIBLE_DEVICES": ""} if phase == "plan" else None,
        )
        print(f"{phase}: complete", flush=True)
    if selected in {"all", "execute"}:
        for index, node in enumerate(nodes[1:], start=1):
            copy_back(node, rank_directories(nodes, index), roots[index])
        result = roots[0] / "result.json"
        if result.exists():
            (root / "result.json").write_text(result.read_text())
