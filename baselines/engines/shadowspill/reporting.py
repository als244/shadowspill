"""Baseline rows and short console summaries derived from the saved Plan."""

from baselines.common.results import host_peak_bytes, memory_peaks

GIB = 1 << 30


def pool_record(value):
    return {name: int(getattr(value, name)) for name, _ in value._fields_}


def failure(error):
    import torch

    from shadowspill.errors import (
        AdmissionError,
        PlanInfeasibleError,
        PlanSearchExhaustedError,
    )

    boundary = isinstance(
        error,
        (
            torch.OutOfMemoryError,
            AdmissionError,
            PlanInfeasibleError,
            PlanSearchExhaustedError,
        ),
    )
    return {
        "status": "cuda_out_of_memory" if boundary else "failed",
        "boundary": boundary,
    }


def memory(all_ranks):
    value = memory_peaks(all_ranks)
    value.update(
        peak_allocated_aggregation="maximum per-rank execution-pool peak occupancy",
        peak_reserved_aggregation="total GPU allowance per rank, including overhead",
        shadowspill=[row["shadowspill_memory"] for row in all_ranks],
        host_peak_aggregation="sum of per-rank planned peak spill bytes",
    )
    value["host_peak_bytes"] = host_peak_bytes(
        {"engine": "shadowspill", "memory": value}
    )
    return value


def describe(case, plan, geometry, selected):
    summary = plan.summary
    seconds = summary["duration_seconds"]
    print(
        f"[shadowspill.plan] rank={case.rank} "
        f"{geometry.sequences_per_microbatch} sequences "
        f"x {geometry.sequence_length} tokens; "
        f"{geometry.accumulation} examples/update; {selected['ordering']}\n"
        f"  predicted {seconds:.6f} s/update, "
        f"{geometry.tokens_per_step / seconds:,.1f} global tokens/s\n"
        f"  GPU peak {summary['execution_peak_bytes'] / GIB:.3f} GiB "
        f"within {plan.memory_budgets.execution_bytes / GIB:g} GiB total; "
        f"spill peak {summary['spill_peak_bytes'] / GIB:.3f} GiB",
        flush=True,
    )


def comparison(plan, measured, rank):
    predicted = plan.summary["duration_seconds"]
    error = 100 * (measured / predicted - 1)
    print(
        f"[shadowspill.measure] rank={rank} median step={measured:.6f}s; "
        f"predicted={predicted:.6f}s; simulator error={error:+.2f}% (warmup excluded)",
        flush=True,
    )
    return {
        "predicted_step_seconds": predicted,
        "median_measured_step_seconds": measured,
        "simulator_error_percent": error,
        "error_definition": "100 * (measured / predicted - 1)",
    }
