"""Repeat a saved distributed search with hypothetical host-copy bandwidth.

This module does not import PyTorch, allocate a GPU, or execute a training step.
It consumes the opt-in search_capture archive and the normal quickstart report.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

from shadowspill.planner import SearchOptions
from shadowspill.planner.program_inputs import (
    ShadowSpillPlanningProblem,
    TransferBandwidths,
)
from shadowspill.search.planner import _Carried, _Planner
from shadowspill.search.refusals import _EXHAUSTED, _INFEASIBLE
from shadowspill.search.report import StepSearchPoint, StepSearchReport

from .matrix import file_digest, write


def host_rates(gigabytes_per_second, measured):
    """Override only copy rates; preserve measured latency and timed phases."""
    rate = round(gigabytes_per_second * 1_000_000_000)
    if rate <= 0:
        raise ValueError("host bandwidth must be positive")
    return TransferBandwidths(
        rate,
        rate,
        rate,
        rate,
        fetch_latency_ns=measured.fetch_latency_ns,
        evict_latency_ns=measured.evict_latency_ns,
    )


def replan(source, output, *, gbps, workers=None, jobs=1):
    if jobs < 1:
        raise ValueError("jobs must be positive")
    source, output = Path(source).resolve(), Path(output).resolve()
    inputs = source / "search-inputs"
    if (inputs / "export-errors.log").exists():
        raise ValueError(f"search input export failed: {inputs / 'export-errors.log'}")
    baseline = StepSearchReport.load(source / "search.json")
    metadata = json.loads((inputs / "metadata.json").read_text())
    if not metadata.get("complete"):
        raise ValueError("the original geometry search has not finished")
    fingerprint = {
        "search_sha256": file_digest(source / "search.json"),
        "inputs_sha256": file_digest(inputs / "metadata.json"),
        "host_bandwidth_GB_per_second": gbps,
        "workers": workers,
        "jobs": jobs,
    }
    output.mkdir(parents=True, exist_ok=True)
    request_path = output / "request.json"
    if request_path.exists() and json.loads(request_path.read_text()) != fingerprint:
        raise ValueError("output directory belongs to a different replan request")
    write(request_path, fingerprint)
    if jobs > 1:
        return _parallel_replan(
            source,
            output,
            baseline,
            metadata,
            fingerprint,
            gbps=gbps,
            workers=workers,
            jobs=jobs,
        )
    points, geometries, selected, winners = [], [], {}, {}
    common_rates = None
    for build in baseline.geometries:
        descriptor = json.loads(
            (inputs / "geometries" / f"{build.step_program_digest}.json").read_text()
        )
        carried = None
        last_problem = None
        for execution, spill in sorted(baseline.budgets):
            request = json.loads(
                (
                    inputs
                    / "requests"
                    / descriptor["local_problem_digest"]
                    / f"{execution}-{spill}.json"
                ).read_text()
            )
            problem_path = inputs / request["problem"]
            if last_problem is None or last_problem.digest != request["problem_digest"]:
                last_problem = ShadowSpillPlanningProblem.from_json(
                    problem_path.read_text()
                )
                if last_problem.digest != request["problem_digest"]:
                    raise ValueError(f"input program identity changed: {problem_path}")
            problem = last_problem
            rates = host_rates(
                gbps, TransferBandwidths.from_value(request["transfer_bandwidths"])
            )
            common_rates = rates
            options = (
                None
                if request["search_options"] is None
                else SearchOptions.from_dict(
                    request["search_options"], "search_options"
                )
            )
            if workers is not None and options is not None:
                options = replace(options, workers=workers)
            ask = _Planner(
                rates,
                options,
                output / "artifacts",
                output / "plans",
                "contribute",
                False,
                True,
            )
            started = time.perf_counter()
            status, failure, makespan, summary = "succeeded", None, None, None
            outcomes, inherited, plan = (), None, None
            try:
                answer = ask.answer(
                    problem,
                    execution,
                    spill,
                    carried if request["incumbents"] else None,
                )
                plan = answer.plan or ask.plan(problem, execution, spill)
            except _EXHAUSTED as error:
                status, failure = "search_exhausted", str(error)
            except _INFEASIBLE as error:
                status, failure = "infeasible", str(error)
            except RuntimeError as error:
                status, failure = "rejected", str(error)
            else:
                makespan, summary = answer.makespan_ns / 1e9, answer.summary
                outcomes = answer.outcomes
                if carried is not None and answer.answered_with_incumbent:
                    inherited = carried.execution_budget_bytes
                if carried is None or answer.makespan_ns < carried.makespan_ns:
                    carried = _Carried(execution, spill, answer.makespan_ns, plan)
                budget = (execution, spill)
                if budget not in selected or answer.makespan_ns < selected[budget]:
                    selected[budget] = answer.makespan_ns
                    destination = output / "winners" / f"{execution}-{spill}.plan.json"
                    destination.parent.mkdir(exist_ok=True)
                    temporary = destination.with_suffix(".tmp")
                    temporary.write_text(plan.to_json() + "\n")
                    temporary.replace(destination)
                    winners[f"{execution}-{spill}"] = {
                        "plan": str(destination.relative_to(output)),
                        "candidate": build.candidate,
                        "ordering": build.ordering.to_dict(),
                        "ordering_label": build.ordering.label,
                        "program_digest": plan.result.program.digest,
                        "schedule_digest": plan.result.schedule.digest,
                        "layout_digest": plan.fixed_layout.digest,
                    }
            points.append(
                StepSearchPoint(
                    build.candidate,
                    build.accumulation_count,
                    build.ordering,
                    execution,
                    spill,
                    status,
                    makespan,
                    summary,
                    failure,
                    time.perf_counter() - started,
                    outcomes,
                    inherited,
                )
            )
            print(
                f"{build.candidate} {build.ordering.label} "
                f"@ {execution / 2**30:.3f} GiB: "
                f"{status}" + (f" {makespan:.6f} s" if makespan is not None else ""),
                flush=True,
            )
            write(output / "progress.json", {"points": len(points), "winners": winners})
        geometries.append(
            replace(
                build,
                build_seconds=0,
                phase_seconds={},
                transfer_bandwidths=common_rates,
            )
        )

    # A microbatch which failed GPU capture has no program to simulate. Preserve
    # its explicit failure instead of claiming the bandwidth change fixes it.
    built = {
        (b.candidate, b.accumulation_count, b.ordering) for b in baseline.geometries
    }
    for point in baseline.points:
        if (point.candidate, point.accumulation_count, point.ordering) not in built:
            if point.status == "succeeded":
                raise ValueError("baseline succeeded without an exported geometry")
            points.append(replace(point, search_seconds=0))
    if len(points) != len(baseline.points):
        raise ValueError("replan does not cover the baseline search matrix")
    report = StepSearchReport(
        baseline.budgets,
        tuple(geometries),
        tuple(points),
        metadata={
            **baseline.metadata,
            "simulation_only": True,
            "source_report": str(source / "search.json"),
            **fingerprint,
        },
        search_options=baseline.search_options,
        # Per-geometry records preserve any different calibrated latencies.
        transfer_bandwidths=None,
    )
    report.save(output / "search.json")
    write(output / "winners.json", winners)
    comparison(output, baseline, report, metadata, gbps)
    write(
        output / "complete.json",
        {
            "status": "simulated",
            "executed_steps": 0,
            "points": len(points),
            "winners": len(winners),
        },
    )
    return report


def _replan_one(arguments):
    source, output, gbps, workers = arguments
    replan(source, output, gbps=gbps, workers=workers)
    return output


def _parallel_replan(
    source, output, baseline, metadata, fingerprint, *, gbps, workers, jobs
):
    """Parallelize whole geometries; keep each budget/incumbent chain serial.

    Each process has its own resumable store. Results merge in the baseline's
    order, preserving deterministic winner tie-breaking regardless of timing.
    """
    arguments, built = [], set()
    for index, build in enumerate(baseline.geometries):
        key = (build.candidate, build.accumulation_count, build.ordering)
        if key in built:
            raise ValueError("duplicate geometry in baseline")
        built.add(key)
        directory = output / "geometry-inputs" / f"{index:04d}"
        directory.mkdir(parents=True, exist_ok=True)
        link = directory / "search-inputs"
        if link.exists():
            if link.resolve() != (source / "search-inputs").resolve():
                raise ValueError("geometry inputs belong to another source")
        else:
            link.symlink_to(source / "search-inputs", target_is_directory=True)
        replace(
            baseline,
            geometries=(build,),
            points=tuple(
                p
                for p in baseline.points
                if (p.candidate, p.accumulation_count, p.ordering) == key
            ),
        ).save(directory / "search.json")
        arguments.append(
            (directory, output / "geometry-search" / f"{index:04d}", gbps, workers)
        )

    points, geometries, selected, winners = [], [], {}, {}
    # spawn avoids inheriting any initialized native worker pools.
    with ProcessPoolExecutor(
        max_workers=jobs, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        for directory in pool.map(_replan_one, arguments):
            partial = StepSearchReport.load(directory / "search.json")
            identities = json.loads((directory / "winners.json").read_text())
            geometries.extend(partial.geometries)
            points.extend(partial.points)
            for point in partial.winners:
                key = f"{point.execution_budget_bytes}-{point.spill_budget_bytes}"
                if key in selected and selected[key] <= point.makespan_seconds:
                    continue
                selected[key] = point.makespan_seconds
                identity = identities[key]
                destination = output / "winners" / f"{key}.plan.json"
                destination.parent.mkdir(exist_ok=True)
                temporary = destination.with_suffix(".tmp")
                shutil.copyfile(directory / identity["plan"], temporary)
                temporary.replace(destination)
                winners[key] = {
                    **identity,
                    "plan": str(destination.relative_to(output)),
                }
            write(output / "progress.json", {"points": len(points), "winners": winners})
    for point in baseline.points:
        if (point.candidate, point.accumulation_count, point.ordering) not in built:
            if point.status == "succeeded":
                raise ValueError("baseline succeeded without an exported geometry")
            points.append(replace(point, search_seconds=0))
    if len(points) != len(baseline.points):
        raise ValueError("replan does not cover the baseline search matrix")
    report = StepSearchReport(
        baseline.budgets,
        tuple(geometries),
        tuple(points),
        metadata={
            **baseline.metadata,
            "simulation_only": True,
            "source_report": str(source / "search.json"),
            **fingerprint,
        },
        search_options=baseline.search_options,
        transfer_bandwidths=None,
    )
    report.save(output / "search.json")
    write(output / "winners.json", winners)
    comparison(output, baseline, report, metadata, gbps)
    write(
        output / "complete.json",
        {
            "status": "simulated",
            "executed_steps": 0,
            "points": len(points),
            "winners": len(winners),
        },
    )
    return report


def comparison(output, baseline, report, metadata, gbps):
    mapping = metadata.get("physical_to_execution_bytes", {})
    physical = {int(value): int(key) for key, value in mapping.items()}
    tokens = metadata.get("global_tokens_per_step")
    lines = [
        f"# Sharded DP at {gbps:g} GB/s",
        "",
        "Simulation only; all host-copy lanes use decimal GB/s. Measured copy latency, "
        "GPU compute, collective timing, overlap costs, shard sizes, and "
        "physical memory "
        "limits are unchanged. Each row independently selects its best geometry, "
        "ordering, resolution and policy.",
        "",
        "| GPU budget (GiB) | Measured-link search (s) | New sim (s) | New tok/s | "
        "New microbatch | Ordering | Recompute | Policy | Plan |",
        "|---:|---:|---:|---:|---|---|---:|---|---|",
    ]
    for execution, spill in report.budgets:
        old, new = baseline.winner(execution, spill), report.winner(execution, spill)
        label = physical.get(execution, execution) / 2**30
        old_s = "—" if old is None else f"{old.makespan_seconds:.3f}"
        if new is None:
            lines.append(
                f"| {label:g} | {old_s} | infeasible | — | — | — | — | — | — |"
            )
            continue
        speed = "—" if not tokens else f"{tokens / new.makespan_seconds:,.0f}"
        summary = {} if new.summary is None else new.summary.as_dict()
        selected = summary.get("selected_candidate", {})
        incumbent = selected.get("incumbent")
        policy = (
            incumbent["found_by"]
            if incumbent is not None
            else "/".join(
                str(selected.get(k, "—")) for k in ("residency_strategy", "fetch_rule")
            )
        )
        fraction = summary.get("recomputing_group_fraction")
        recompute = "—" if fraction is None else f"{fraction:.0%}"
        lines.append(
            f"| {label:g} | {old_s} | {new.makespan_seconds:.3f} | {speed} | "
            f"{new.candidate} | {new.ordering.label} | {recompute} | {policy} | "
            f"[plan](winners/{execution}-{spill}.plan.json) |"
        )
    lines += [
        "",
        "[Every search point and resolution](search.json) · "
        "[Winner identities](winners.json) · [Request](request.json)",
        "",
    ]
    (output / "RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Concurrent geometries; budgets stay ordered within each (default: 1)",
    )
    parser.add_argument(
        "source", type=Path, help="quickstart rank directory with search-inputs/"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--host-gbps",
        type=float,
        required=True,
        help="decimal GB/s in each direction, alone and simultaneous",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="CPU planner workers (default: 4)",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Save the normal search figures and raw plot data",
    )
    args = parser.parse_args()
    report = replan(
        args.source,
        args.output,
        gbps=args.host_gbps,
        workers=args.workers,
        jobs=args.jobs,
    )
    if args.plots:
        from shadowspill.plots import plot_step_search

        plot_step_search(report, args.output / "figures")


if __name__ == "__main__":
    main()
