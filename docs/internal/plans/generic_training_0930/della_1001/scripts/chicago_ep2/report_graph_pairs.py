"""Export readable graph-pair tables from completed experiment diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def report(path: Path) -> None:
    diagnostics = json.loads(path.read_text())
    rows = []
    for stage in diagnostics["unique_stages"]:
        for pair in stage["graph_pairs"]:
            for direction in ("forward", "backward"):
                graph = pair[direction]
                if graph is None:
                    continue
                sizes = {
                    "input": graph["input_allocation_bytes"],
                    "mutated": graph["mutation_allocation_bytes"],
                    "output": graph["output_allocation_bytes"],
                    "workspace": graph["task_workspace_bytes"],
                }
                allocations = {
                    obj["alias_group_id"]: obj["allocation_size_bytes"]
                    for category in ("inputs", "mutations", "outputs")
                    for obj in graph[category]
                }
                rows.append(
                    {
                        "stage": stage["unique_stage_id"],
                        "modules": ",".join(stage["module_targets"]),
                        "variant": pair["variant"],
                        "direction": direction,
                        "runtime_ms": graph["runtime_ns"] / 1e6,
                        **{f"{name}_gib": size / 2**30 for name, size in sizes.items()},
                        "category_sum_gib": sum(sizes.values()) / 2**30,
                        "unique_objects_plus_workspace_gib": (
                            sum(allocations.values()) + sizes["workspace"]
                        )
                        / 2**30,
                    }
                )
    if not rows:
        raise ValueError(f"No graph pairs in {path}")
    target = path.with_name("graph-pairs.csv")
    with target.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    columns = list(rows[0])
    lines = [
        "# Graph-pair profiles",
        "",
        "Sizes are GiB per rank. Category sums can count aliases more than once;",
        "the final column deduplicates object storage across categories. Neither",
        "column is a measured whole-step peak. External communication allocations",
        "are recorded separately in communication-memory.json.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                f"{value:.5f}"
                if isinstance(value, float)
                else str(value).replace("|", "\\|")
                for value in row.values()
            )
            + " |"
        )
    target.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print(f"{target}: {len(rows)} graph profiles", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", type=Path)
    args = parser.parse_args()
    for path in sorted(args.experiment.rglob("plan-diagnostics.json")):
        report(path)


if __name__ == "__main__":
    main()
