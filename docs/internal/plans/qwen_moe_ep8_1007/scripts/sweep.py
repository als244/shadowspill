"""Resume Qwen EP8 quickstart: search all geometries, then measure winners.

Every subprocess runs in the invoking tmux pane. Completed cases are skipped;
an interrupted attempt keeps its logs and contributes reusable build/plan stores.
"""

import argparse
import csv
from dataclasses import fields
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "workloads").is_dir())
sys.path.insert(0, str(ROOT))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def now():
    return datetime.now(UTC).isoformat()


def run_command(command, log_path):
    with log_path.open("w") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def execute_case(args, name, model, tokens, *, run_budgets=(), tiny=False):
    directory = args.outdir / name
    complete = directory / "completed.json"
    if complete.exists():
        print(f"[{now()}] SKIP {name}: already complete", flush=True)
        return json.loads(complete.read_text())["attempt"]
    attempts = directory / "attempts"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    attempt = attempts / stamp
    attempt.mkdir(parents=True, exist_ok=False)
    store = args.outdir / "stores" / model / ("smoke" if tiny else f"t{tokens}")
    command = [
        sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
        "--nproc-per-node", str(args.ranks), "--tee", "3",
        "--log-dir", str(attempt / "processes"), str(HERE / "experiment.py"),
        "--model", model, "--tokens-per-rank", str(tokens),
        "--global-tokens", str(args.ranks * tokens * 2 if tiny else args.global_tokens),
        "--sequence-length", "64" if tiny else "1024",
        "--search-budgets", "6" if tiny else ",".join(map(str, args.budgets)),
        "--run-budgets", "6" if tiny else ",".join(map(str, run_budgets)),
        "--spill-gib", "4" if tiny else str(args.spill_gib),
        "--external-headroom-gib", "0.5" if tiny else "4",
        "--steps", "2" if tiny else str(args.steps),
        "--outdir", str(attempt / "report"),
        "--artifact-store", str(store / "build"), "--plan-store", str(store / "plans"),
        "--search-workers", str(args.search_workers),
        "--symmetric-planning" if args.symmetric_planning else "--no-symmetric-planning",
    ]
    if tiny:
        command += ["--tiny", "--no-plots", "--profile-conditioning-seconds", "0",
                    "--profile-measurement-seconds", "0"]
    record = {"case": name, "model": model, "tokens_per_rank": tokens,
              "started": now(), "status": "running", "command": command,
              "attempt": str(attempt), "slurm_job_id": os.getenv("SLURM_JOB_ID")}
    write(directory / "status.json", record)
    print(f"[{now()}] START {name}", flush=True)
    started = time.monotonic()
    code = run_command(command, attempt / "console.log")
    done = all((attempt / "report" / f"rank-{rank:05d}" / "completed.json").exists()
               for rank in range(args.ranks))
    record.update(finished=now(), seconds=time.monotonic() - started, exit_code=code,
                  status="passed" if code == 0 and done else "failed")
    write(directory / "status.json", record)
    print(f"[{now()}] END {name}: {record['status']} ({record['seconds']:.1f}s)", flush=True)
    if record["status"] != "passed":
        raise RuntimeError(f"{name} failed; inspect {attempt / 'console.log'}")
    write(complete, record)
    return str(attempt)


def combine(args, model):
    from shadowspill.pytorch import StepSearchReport
    from shadowspill.plots import plot_step_search

    completed = [args.outdir / "search" / model / f"t{tokens}" / "completed.json"
                 for tokens in args.tokens]
    if not all(path.exists() for path in completed):
        return None
    by_rank = []
    for rank in range(args.ranks):
        reports = [StepSearchReport.load(
            Path(json.loads(path.read_text())["attempt"]) / "report" / f"rank-{rank:05d}" / "search.json"
        ) for path in completed]
        if any(report.budgets != reports[0].budgets for report in reports):
            raise RuntimeError("Geometry reports have different resolved budgets; inspect carve-outs before merging")
        merged = StepSearchReport(
            budgets=reports[0].budgets,
            geometries=tuple(g for report in reports for g in report.geometries),
            points=tuple(p for report in reports for p in report.points),
            search_options=reports[0].search_options,
            metadata={**reports[0].metadata,
                      "model": model, "rank": rank, "world_size": args.ranks,
                      "global_tokens_per_step": args.global_tokens,
                      "tokens_per_microbatch_per_rank": list(args.tokens),
                      "microbatches_per_rank": [args.global_tokens // (args.ranks * tokens)
                                                for tokens in args.tokens],
                      "sources": [str(path) for path in completed]},
        )
        root = args.outdir / "combined" / model / f"rank-{rank:05d}"
        root.mkdir(parents=True, exist_ok=True)
        merged.save(root / "search.json")
        plot_step_search(merged, root / "figures")
        by_rank.append(merged)
    winners = []
    for requested, budget in zip(args.budgets, by_rank[0].budgets, strict=True):
        local = [report.winner(*budget) for report in by_rank]
        identities = [(p.candidate, p.ordering.label) if p else None for p in local]
        if any(item != identities[0] for item in identities):
            raise RuntimeError(f"Ranks disagree on winner at {requested} GiB: {identities}")
        if local[0] is None:
            winners.append({"budget_gib": requested, "status": "infeasible"})
            continue
        winner = local[0]
        winners.append({"budget_gib": requested, "status": "succeeded",
                        "tokens_per_rank": int(winner.candidate),
                        "ordering": winner.ordering.to_dict(),
                        "ordering_label": winner.ordering.label,
                        "predicted_seconds": max(p.makespan_seconds for p in local)})
    write(args.outdir / "combined" / model / "winners.json", winners)
    return winners


def read_measurements(directory):
    """Read quickstart's durable tables without loading a model or a plan."""
    from shadowspill.plots import RunBudgetOutcome

    def budget_bytes(value):
        return round(float(value) * (1 << 30))

    steps = {}
    with (directory / "steps.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            steps.setdefault(budget_bytes(row["execution_budget_gib"]), []).append(
                (int(row["step"]), float(row["seconds"]))
            )
    outcomes = []
    with (directory / "run_budgets.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            budget = budget_bytes(row["execution_budget_gib"])
            values = {
                field.name: float(row[field.name]) if row[field.name] else None
                for field in fields(RunBudgetOutcome)
                if field.name not in {"execution_budget_bytes", "step_seconds"}
            }
            outcomes.append(RunBudgetOutcome(
                execution_budget_bytes=budget,
                step_seconds=tuple(value for _, value in sorted(steps.get(budget, ()))),
                **values,
            ))
    return outcomes


def combine_measurements(args, model, winners):
    """Put all budget winners in one report, with rank-local and slowest times."""
    from shadowspill.plots import plot_step_run
    from shadowspill.pytorch import StepSearchReport

    root = args.outdir / "combined" / model
    feasible = [row for row in winners if row["status"] == "succeeded"]
    by_rank = []
    for rank in range(args.ranks):
        report = StepSearchReport.load(root / f"rank-{rank:05d}" / "search.json")
        expected = {
            budget[0] for row, budget in zip(winners, report.budgets, strict=True)
            if row["status"] == "succeeded"
        }
        outcomes = {}
        for tokens in sorted({row["tokens_per_rank"] for row in feasible}):
            complete = args.outdir / "measure" / model / f"t{tokens}" / "completed.json"
            if not complete.exists():
                raise RuntimeError(f"Measure all quickstart winners first: {complete}")
            attempt = Path(json.loads(complete.read_text())["attempt"])
            tables = attempt / "report" / f"rank-{rank:05d}" / "figures" / "raw_data"
            for outcome in read_measurements(tables):
                if outcome.execution_budget_bytes in outcomes:
                    raise RuntimeError(f"Duplicate measured budget in {tables}")
                outcomes[outcome.execution_budget_bytes] = outcome
        if outcomes.keys() != expected:
            raise RuntimeError(f"Measured budgets differ from winners on rank {rank}")
        if outcomes:
            plot_step_run(list(outcomes.values()), root / f"rank-{rank:05d}" / "figures",
                          units_per_step=args.global_tokens // args.ranks, unit_label="tokens")
        by_rank.append(outcomes)
    measured = []
    for row, budget in zip(winners, report.budgets, strict=True):
        if row["status"] != "succeeded":
            continue
        seconds = [outcomes[budget[0]].measured_step_seconds for outcomes in by_rank]
        measured.append({**row, "measured_seconds_by_rank": seconds,
                         "measured_seconds": max(seconds),
                         "global_tokens_per_second": args.global_tokens / max(seconds)})
    write(root / "measured-winners.json", measured)
    return measured


def train_winner(args, model, winners):
    directory = args.outdir / "training" / model
    if (directory / "completed.json").exists():
        print(f"[{now()}] SKIP completed real-data training for {model}", flush=True)
        return
    feasible = [row for row in winners if row["status"] == "succeeded"]
    if not feasible:
        raise RuntimeError(f"No feasible training plan for {model}")
    # Quickstart measurement must finish before starting the real-data run.
    for row in feasible:
        completion = args.outdir / "measure" / model / f"t{row['tokens_per_rank']}" / "completed.json"
        if not completion.exists():
            raise RuntimeError(f"Measure all quickstart winners before training: {completion}")
    selected = min(feasible, key=lambda row: row["measured_seconds"])
    data = args.outdir.parent / "data" / model
    if not (data / "meta.json").exists():
        raise RuntimeError(f"Prepare tokenizer/data on head node first: {data}")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    attempt = directory / "attempts" / stamp
    attempt.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone",
               "--nproc-per-node", str(args.ranks), "--tee", "3",
               "--log-dir", str(attempt / "processes"), str(HERE / "train.py"),
               "--model", model, "--tokens-per-rank", str(selected["tokens_per_rank"]),
               "--execution-gib", str(selected["budget_gib"]),
               "--spill-gib", str(args.spill_gib), "--global-tokens", str(args.global_tokens),
               "--breadth", str(selected["ordering"]["breadth"]),
               "--steps", str(args.training_steps), "--data", str(data),
               "--outdir", str(attempt / "run"),
               "--artifact-store", str(args.outdir / "stores" / model / "training"),
               "--symmetric-planning" if args.symmetric_planning else "--no-symmetric-planning"]
    record = {"status": "running", "started": now(), "command": command,
              "selected": selected, "attempt": str(attempt)}
    write(directory / "status.json", record)
    print(f"[{now()}] START real-data training {model}: {selected}", flush=True)
    code = run_command(command, attempt / "console.log")
    complete = code == 0 and all((attempt / "run" / f"rank-{rank:05d}" / "completed.json").exists()
                                for rank in range(args.ranks))
    record.update(status="passed" if complete else "failed", finished=now(), exit_code=code)
    write(directory / "status.json", record)
    if not complete:
        raise RuntimeError(f"Training failed: {attempt / 'console.log'}")
    write(directory / "completed.json", record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=Path.home() / "storage/shadowspill/qwen_moe_ep8_1007/ep8-bf16")
    parser.add_argument("--stage", choices=("smoke", "search", "measure", "train", "all"), default="all")
    parser.add_argument("--ranks", type=int, default=8)
    parser.add_argument("--models", nargs="+", default=["qwen3moe", "qwen35moe"])
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 16384, 32768, 65536])
    parser.add_argument("--budgets", type=int, nargs="+", default=[20, 30, 40, 50, 60, 70])
    parser.add_argument("--global-tokens", type=int, default=1 << 22)
    parser.add_argument("--spill-gib", type=float, default=64)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--training-steps", type=int, default=10)
    parser.add_argument("--search-workers", type=int, default=4)
    parser.add_argument("--symmetric-planning", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not os.getenv("SLURM_JOB_ID"):
        parser.error("run in the allocated codex GPU pane")
    if args.ranks <= 0 or args.steps < 3 or args.training_steps <= 0 or args.search_workers < 0:
        parser.error("ranks/training steps must be positive, measured steps >= 3, workers >= 0")
    if not args.tokens or any(tokens <= 0 or tokens % 1024 or
                             args.global_tokens % (args.ranks * tokens) for tokens in args.tokens):
        parser.error("token counts must divide the global batch and contain whole 1K sequences")
    if not args.budgets or any(budget <= 0 for budget in args.budgets) or args.spill_gib <= 0:
        parser.error("execution and spill budgets must be positive")
    if len(set(args.tokens)) != len(args.tokens) or len(set(args.budgets)) != len(args.budgets):
        parser.error("token counts and budgets must not repeat")
    args.outdir.mkdir(parents=True, exist_ok=True)
    configuration = {key: str(value) if isinstance(value, Path) else value
                     for key, value in vars(args).items() if key != "stage"}
    configuration_path = args.outdir / "configuration.json"
    if configuration_path.exists():
        previous = json.loads(configuration_path.read_text())
        previous.pop("stage", None)
        if previous != configuration:
            parser.error("sweep settings differ from this output directory; use a new --outdir")
    else:
        write(configuration_path, configuration)
    if args.stage in {"all", "smoke"}:
        for model in args.models:
            execute_case(args, f"smoke/{model}", model, 256, tiny=True)
    if args.stage in {"all", "search"}:
        # Alternate models so progress covers both architectures at low tokens.
        for tokens in args.tokens:
            for model in args.models:
                execute_case(args, f"search/{model}/t{tokens}", model, tokens)
    all_winners = {}
    for model in args.models:
        winners = combine(args, model)
        all_winners[model] = winners
        if args.stage not in {"all", "measure"} or winners is None:
            continue
        for tokens in args.tokens:
            budgets = [row["budget_gib"] for row in winners
                       if row.get("tokens_per_rank") == tokens]
            if budgets:
                execute_case(args, f"measure/{model}/t{tokens}", model, tokens, run_budgets=budgets)
    if args.stage in {"all", "measure", "train"}:
        for model, winners in all_winners.items():
            if winners is None:
                raise RuntimeError(f"Complete the quickstart searches before measuring {model}")
            all_winners[model] = combine_measurements(args, model, winners)
    if args.stage in {"all", "train"}:
        for model in args.models:
            if all_winners[model] is None:
                raise RuntimeError(f"Complete the quickstart searches before training {model}")
            train_winner(args, model, all_winners[model])


if __name__ == "__main__":
    main()
