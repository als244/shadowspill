"""CPU-only phase/device report, intended to run after releasing the GPUs."""

import argparse
import csv
import json
import statistics
from datetime import UTC, datetime
from pathlib import Path


def spans(events):
    opened, result = {}, []
    for event in events:
        key = event["pid"], event["name"]
        if event["boundary"] == "begin":
            if key in opened:
                raise ValueError(f"phase opened twice: {key}")
            opened[key] = event
        elif event["boundary"] == "end":
            start = opened.pop(key)
            result.append(
                dict(
                    name=event["name"],
                    rank=event["rank"],
                    hostname=event["hostname"],
                    start_ns=start["timestamp_ns"],
                    end_ns=event["timestamp_ns"],
                    seconds=(event["monotonic_ns"] - start["monotonic_ns"]) / 1e9,
                    error=event.get("error"),
                )
            )
    return result, list(opened.values())


def devices(path):
    result = []
    with path.open() as stream:
        for raw in csv.DictReader(stream):
            row = {k.strip().split(" [")[0]: v.strip() for k, v in raw.items()}
            if row["timestamp"] == "timestamp":
                continue
            at = datetime.strptime(row["timestamp"], "%Y/%m/%d %H:%M:%S.%f").replace(
                tzinfo=UTC
            )
            record = dict(timestamp_ns=int(at.timestamp() * 1e9), uuid=row["uuid"])
            for key in ("utilization.gpu", "memory.used", "power.draw"):
                try:
                    record[key] = float(row[key].split()[0])
                except (ValueError, KeyError):
                    record[key] = None
            result.append(record)
    return result


def metrics(samples, start, end):
    rows = [row for row in samples if start <= row["timestamp_ns"] < end]
    utilization = [
        r["utilization.gpu"] for r in rows if r["utilization.gpu"] is not None
    ]
    memory = [r["memory.used"] for r in rows if r["memory.used"] is not None]
    power = [r["power.draw"] for r in rows if r["power.draw"] is not None]
    return dict(
        samples=len(rows),
        gpu_count=len({row["uuid"] for row in rows}),
        sampled_gpu_util_percent_mean=statistics.mean(utilization)
        if utilization
        else None,
        peak_sampled_memory_mib=max(memory) if memory else None,
        sampled_power_watts_mean=statistics.mean(power) if power else None,
    )


def report(directory):
    root = Path(directory)
    result_path = root / "result.json"
    result = json.loads(result_path.read_text()) if result_path.exists() else None
    phases, unfinished, observations = [], [], []
    journals = sorted(root.glob("rank-*/phases.jsonl"))
    if (root / "launcher.jsonl").exists():
        journals.insert(0, root / "launcher.jsonl")
    for path in journals:
        events = [json.loads(line) for line in path.read_text().splitlines()]
        rows, opened = spans(events)
        unfinished.extend(opened)
        metric_path = path.parent / "device-metrics.csv"
        # A parent benchmark monitor observes all ranks on its node; a custom
        # torchrun script may instead have one monitor in each rank directory.
        if not metric_path.exists():
            metric_path = root / "device-metrics.csv"
        samples = devices(metric_path) if metric_path.exists() else []
        for row in rows:
            row["device"] = metrics(samples, row["start_ns"], row["end_ns"])
        phases.extend(rows)
    for path in sorted(root.glob("**/monitor.json")):
        observations.append(
            dict(file=str(path.relative_to(root)), **json.loads(path.read_text()))
        )
    return dict(
        result=result,
        phases=phases,
        unfinished=unfinished,
        monitors=observations,
        notes=[
            "Phase durations use the originating host's monotonic clock; "
            "device timestamps are UTC.",
            "Nested phases must not be added. Trainer.step seconds exclude "
            "input production, metric reducers and logging.",
            "Device utilization is sampled activity, not achieved FLOPs. "
            "Missing samples remain missing, not zero.",
            "One-second device sampling can miss short phases and memory peaks. "
            "No cross-host clock agreement is assumed.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    args = parser.parse_args()
    path = Path(args.directory) / "phase-report.json"
    path.write_text(json.dumps(report(args.directory), indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
