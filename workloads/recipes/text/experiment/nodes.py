"""Homogeneous ranks across several machines: one Plan, built once, run everywhere.

``nodes`` beside ``ranks`` lists the machines. The first builds, profiles and
plans as a single-node run would; the artifacts the grouped phases need are
copied to the other nodes, which must hold the same hardware, since every
rank runs the one Plan; then the communicate and execute phases launch every
node's workers under one rendezvous, and the remote ranks' directories are
copied back afterwards.
"""

from __future__ import annotations

from workloads.recipes.phases import run_phases
from workloads.recipes.text.launch import (
    check_nodes,
    copy_back,
    copy_to,
    node_ranks,
    rank_directories,
    run_phase,
)

PACKAGE = "workloads.recipes.text.experiment"
PHASES = ("build", "profile", "communicate", "plan", "execute")
#: What a grouped phase on another node reads.
SHARED = {
    "communicate": ("config.json", "candidates.json", "programs"),
    "execute": ("config.json", "planning.json", "plans"),
}


def run_nodes(root, raw, selected, nodes):
    check_nodes(nodes)
    if raw.get("ranks", sum(node_ranks(nodes))) != sum(node_ranks(nodes)):
        raise ValueError("ranks must equal the nodes' ranks added up")
    logs = root / "logs"
    for phase in PHASES:
        if selected not in {"all", phase}:
            continue
        if phase in SHARED:
            for node in nodes[1:]:
                copy_to(node, SHARED[phase], root)
            print(
                f"{phase}: starting on {len(nodes)} nodes; logs {logs}/{phase}*.log",
                flush=True,
            )
            run_phase(
                nodes,
                f"{PACKAGE}.{phase}",
                [root] * len(nodes),
                logs,
                phase,
                grouped=True,
                port=raw.get("port", 29400),
            )
            print(f"{phase}: complete", flush=True)
        else:
            run_phases(PACKAGE, root, phase)
    if selected in {"all", "execute"}:
        for index, node in enumerate(nodes[1:], start=1):
            copy_back(node, rank_directories(nodes, index), root)
