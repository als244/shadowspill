"""Validate both ranks, five full updates, annotations and device metrics."""

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path


def audit(directory):
    connection = sqlite3.connect(f"file:{directory / 'trace.sqlite'}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    tables = {
        r[0]
        for r in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    required = {"NVTX_EVENTS", "CUPTI_ACTIVITY_KIND_KERNEL", "GPU_METRICS", "StringIds"}
    if missing := required - tables:
        raise ValueError(f"Missing trace tables: {sorted(missing)}")
    strings = dict(connection.execute("SELECT id,value FROM StringIds"))
    labels = Counter()
    steps = defaultdict(list)
    processes = {}
    unfinished_steps = []
    for row in connection.execute(
        "SELECT start,end,text,textId,globalTid FROM NVTX_EVENTS"
    ):
        label = row["text"] or strings.get(row["textId"], "")
        labels[label] += 1
        match = re.fullmatch(r"ep2/training/step_(\d+)/rank_(\d+)", label)
        if not match:
            continue
        step, rank = map(int, match.groups())
        if row["end"] is None:
            unfinished_steps.append(label)
            continue
        pid = row["globalTid"] & ~((1 << 24) - 1)
        processes[pid] = rank
        steps[pid].append(
            dict(
                rank=rank,
                step=step,
                start_ns=row["start"],
                end_ns=row["end"],
                seconds=(row["end"] - row["start"]) / 1e9,
            )
        )
    for rows in steps.values():
        rows.sort(key=lambda r: r["step"])
    expected = {}
    for path in directory.glob("rank-*/profile-result.json"):
        result = json.loads(path.read_text())
        expected[result["rank"]] = [
            r["step"] for r in result["updates"] if r["kind"] == "training"
        ]
    errors = []
    if len(processes) != 2 or set(processes.values()) != {0, 1}:
        errors.append(f"Expected two worker processes; saw {processes}")
    for pid, rank in processes.items():
        if [r["step"] for r in steps[pid]] != expected.get(rank):
            errors.append(f"Rank {rank} step ranges do not match completed updates")
    if unfinished_steps:
        errors.append(f"Unfinished training ranges: {unfinished_steps}")
    coverage = {}
    devices = Counter()
    outside = []
    for table in (
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMSET",
    ):
        if table not in tables:
            continue
        counts = Counter()
        for row in connection.execute(
            f"SELECT start,end,globalPid,deviceId FROM {table}"
        ):
            counts["activities"] += 1
            if table.endswith("_KERNEL"):
                devices[row["deviceId"]] += 1
            enclosed = any(
                r["start_ns"] <= row["start"] and row["end"] <= r["end_ns"]
                for r in steps.get(row["globalPid"], ())
            )
            counts["inside_training_step"] += int(enclosed)
            if not enclosed and len(outside) < 20:
                outside.append(dict(table=table, **dict(row)))
        coverage[table] = dict(counts)
        if counts["activities"] != counts["inside_training_step"]:
            errors.append(f"GPU activities outside complete update ranges in {table}")
    if len(devices) != 2:
        errors.append(f"Expected kernels on two GPUs; saw {dict(devices)}")
    metrics = [
        dict(r)
        for r in connection.execute(
            "SELECT typeId,COUNT(*) AS samples,COUNT(DISTINCT metricId) AS metrics "
            "FROM GPU_METRICS GROUP BY typeId"
        )
    ]
    if len(metrics) < 2 or any(r["samples"] == 0 for r in metrics):
        errors.append("GPU device metric samples missing for one or both GPUs")
    prefixes = (
        "shadowspill.compiled_call.",
        "shadowspill.before_task.",
        "shadowspill.after_task.",
        "moon_quack/",
        "shadowspill.",
    )
    annotations = {
        prefix: sum(count for name, count in labels.items() if name.startswith(prefix))
        for prefix in prefixes
    }
    if any(annotations[prefix] == 0 for prefix in prefixes):
        errors.append("Required ShadowSpill or Quack annotation ranges are missing")
    report = dict(
        status="FAIL" if errors else "PASS",
        errors=errors,
        steps=[row for rows in steps.values() for row in rows],
        kernel_devices=dict(devices),
        gpu_activity_coverage=coverage,
        outside_step_examples=outside,
        gpu_metrics=metrics,
        annotation_counts=annotations,
        annotation_examples=[
            name for name in labels if name.startswith("shadowspill.")
        ][:30],
        report=str(directory / "trace.nsys-rep"),
    )
    connection.close()
    (directory / "trace-audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    audit(parser.parse_args().directory.resolve())
