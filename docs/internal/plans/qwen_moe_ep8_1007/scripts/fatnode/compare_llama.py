"""Summarize completed Llama DP4 symmetric and independent planning runs."""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def summarize(root):
    ranks = sorted(root.glob("rank-*"))
    results = [read(rank / "result.json") for rank in ranks]
    assert len(results) == 4 and all(row["passed"] for row in results)
    hashes = [read(rank / "parameter_hashes.json") for rank in ranks]
    assert all(value == hashes[0] for value in hashes)
    steps = results[0]["steps"]
    global_losses = [row["global_loss"] for row in steps]
    assert all(
        [row["global_loss"] for row in result["steps"]] == global_losses
        for result in results
    )
    plan = read(ranks[0] / "plan.json")
    prediction = plan["prediction"]["makespan_ns"] / 1e9
    times = [row["slowest_rank_seconds"] for row in steps]
    events = [
        json.loads(line)
        for line in (ranks[0] / "progress.jsonl").read_text().splitlines()
    ]
    prep = next(row for row in events if row["event"] == "prepared")
    prepared_start = next(row for row in events if row["event"] == "prepare_start")
    search = read(ranks[0] / "search.json")
    return dict(
        path=str(root),
        parameters=results[0]["parameters"],
        symmetric_planning=results[0]["symmetric_planning"],
        verified_replica_equality=True,
        global_losses=global_losses,
        historical_dp1_losses=[row["historical_dp1_loss"] for row in steps],
        changed_parameter_tensors=results[0]["changed_parameter_tensors"],
        selected_candidate=results[0]["selected_candidate"],
        save_recompute_counts=dict(
            Counter(row["option_id"] for row in plan["selections"])
        ),
        prepare_seconds=prep["elapsed_s"] - prepared_start["elapsed_s"],
        predicted_step_seconds=prediction,
        median_step_seconds=statistics.median(times),
        step_seconds=times,
        global_tokens_per_second=32768 / statistics.median(times),
        search_points=len(search["points"]),
        search_owners=results[0]["owners"],
        parameter_hashes=hashes[0],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symmetric", type=Path)
    parser.add_argument("independent", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    symmetric, independent = summarize(args.symmetric), summarize(args.independent)
    assert symmetric["symmetric_planning"] and not independent["symmetric_planning"]
    assert (
        symmetric["parameter_hashes"].keys() == independent["parameter_hashes"].keys()
    )
    differing = [
        name
        for name, value in symmetric["parameter_hashes"].items()
        if value != independent["parameter_hashes"][name]
    ]
    differences = [
        a - b
        for a, b in zip(
            symmetric["global_losses"], independent["global_losses"], strict=True
        )
    ]
    for result in (symmetric, independent):
        result.pop("parameter_hashes")
    report = dict(
        symmetric=symmetric,
        independent=independent,
        loss_differences=differences,
        maximum_absolute_loss_difference=max(map(abs, differences)),
        final_weights_bitwise_equal=not differing,
        differing_parameter_tensors=differing,
        note=(
            "Historical DP1 used an earlier software tree and a different "
            "accumulation order."
        ),
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
