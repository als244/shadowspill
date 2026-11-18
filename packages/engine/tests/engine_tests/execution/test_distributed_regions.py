"""Two-device saved training composes DP with checkpoint and execution policies."""

import json
from collections import defaultdict

import pytest
import torch

from engine_tests.execution.test_distributed_executor import (
    test_distributed_saved_executor as _verify,
)


def _check_collective_traces(directory, invocation=0):
    signatures = []
    records_by_rank = []
    for rank in range(2):
        trace = json.loads(
            (directory / f"rank-{rank}-trace-{invocation}.json").read_text()
        )
        assert trace["summary"]["trace_complete"]
        assert not trace["allocator"]["overflow"]
        runtime = trace["runtime"]
        for field in (
            "event_overflow",
            "allocation_event_overflow",
            "callback_failures_after",
            "queued_actions_after",
            "pending_retirements_after",
        ):
            assert not runtime[field], (rank, field)
        collectives = trace["collectives"]
        assert collectives["enabled"] and collectives["complete"]
        for field in ("dropped", "errors", "missing_timings"):
            assert collectives[field] == 0, (rank, field)
        assert all(
            copy["finished_at_ns"] >= copy["started_at_ns"]
            for copy in collectives["host_copies"]["records"]
        )
        records = collectives["records"]
        assert records
        sequences = defaultdict(list)
        for record in records:
            sequences[tuple(record["group_ranks"])].append(record["group_sequence"])
            assert 0 < record["started_at_ns"] <= record["finished_at_ns"]
        for sequence in sequences.values():
            assert sequence == sorted(set(sequence))
        keys = (
            "operation",
            "group_ranks",
            "group_sequence",
            "input_bytes",
            "output_bytes",
            "source",
            "task_id",
            "execution_task_id",
        )
        signatures.append([[record[key] for key in keys] for record in records])
        records_by_rank.append(records)
    assert signatures[0] == signatures[1], "collective order/geometry differs by rank"
    for left, right in zip(*records_by_rank, strict=True):
        assert {left["group_rank"], right["group_rank"]} == {0, 1}


@pytest.mark.cuda
@pytest.mark.parametrize(
    "kind,variant,storage,lora,retain_sequence,nested,checkpoint_cuts",
    [
        pytest.param(
            "step",
            "save",
            "replicated",
            False,
            False,
            False,
            False,
            id="full-replicated-save",
        ),
        pytest.param(
            "step",
            "recompute",
            "optimizer",
            False,
            False,
            False,
            False,
            id="full-optimizer-recompute",
        ),
        pytest.param(
            "step",
            "recompute",
            "gradient_optimizer",
            True,
            True,
            False,
            False,
            id="lora-sharded-grad-retained",
        ),
        pytest.param(
            "step",
            "recompute",
            "sharded",
            False,
            True,
            True,
            False,
            id="full-sharded-composed-retained",
        ),
        pytest.param(
            "step",
            "auto",
            "cutoff",
            True,
            False,
            True,
            True,
            id="lora-cutoff-composed-custom",
        ),
        pytest.param(
            "forward_backward",
            "recompute",
            "sharded",
            True,
            True,
            False,
            False,
            id="lora-sharded-accumulation",
        ),
    ],
)
def test_dp_checkpoint_regions_chunks_and_retention(
    tmp_path, kind, variant, storage, lora, retain_sequence, nested, checkpoint_cuts
):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two distinct GPUs")
    (tmp_path / "regions.json").write_text(
        json.dumps(
            dict(
                storage=storage,
                lora=lora,
                retain_sequence=retain_sequence,
                nested=nested,
                checkpoint_cuts=checkpoint_cuts,
            ),
            indent=2,
        )
        + "\n"
    )
    _verify(tmp_path, kind, variant, "regions")
    _check_collective_traces(tmp_path)


@pytest.mark.cuda
@pytest.mark.parametrize("gate_dtype", ["bfloat16", "float16"])
def test_dp_native_pointwise_emulation(tmp_path, gate_dtype):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two distinct GPUs")
    (tmp_path / "regions.json").write_text(
        json.dumps(
            dict(
                storage="sharded",
                lora=True,
                retain_sequence=True,
                nested=True,
                checkpoint_cuts=False,
                gate_dtype=gate_dtype,
            ),
            indent=2,
        )
        + "\n"
    )
    _verify(tmp_path, "step", "recompute", "regions")
    _check_collective_traces(tmp_path)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "kind,storage,lora,retain_sequence,nested,lifetime_mode",
    [
        pytest.param(
            "step",
            "replicated",
            False,
            False,
            False,
            "move",
            id="full-replicated-movement",
        ),
        pytest.param(
            "step",
            "sharded",
            False,
            False,
            True,
            "replay_compact",
            id="full-sharded-composed-replay-compaction",
        ),
        pytest.param(
            "step",
            "gradient_optimizer",
            True,
            True,
            False,
            "replay_compact",
            id="lora-retained-replay-compaction",
        ),
        pytest.param(
            "forward_backward",
            "sharded",
            True,
            True,
            True,
            "replay_compact",
            id="lora-continuing-composed-replay-compaction",
        ),
    ],
)
def test_dp_public_execution_lifetimes(
    tmp_path, kind, storage, lora, retain_sequence, nested, lifetime_mode
):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two distinct GPUs")
    (tmp_path / "regions.json").write_text(
        json.dumps(
            dict(
                storage=storage,
                lora=lora,
                retain_sequence=retain_sequence,
                nested=nested,
                checkpoint_cuts=False,
                lifetime_mode=lifetime_mode,
            ),
            indent=2,
        )
        + "\n"
    )
    _verify(tmp_path, kind, "recompute", "regions")
    _check_collective_traces(tmp_path)


@pytest.mark.cuda
@pytest.mark.parametrize(
    "kind,lora,retain_sequence",
    [
        pytest.param("step", False, True, id="full-nested-retained"),
        pytest.param("forward_backward", True, False, id="lora-nested-continuing"),
    ],
)
def test_dp_independent_batch_axis(tmp_path, kind, lora, retain_sequence):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two distinct GPUs")
    (tmp_path / "regions.json").write_text(
        json.dumps(
            dict(
                storage="sharded",
                lora=lora,
                retain_sequence=retain_sequence,
                nested=True,
                checkpoint_cuts=False,
                lifetime_mode="replay_compact",
                chunk_axis="batch",
                batch=13,
                sequence=17,
            ),
            indent=2,
        )
        + "\n"
    )
    _verify(tmp_path, kind, "recompute", "regions")
    _check_collective_traces(tmp_path)


@pytest.mark.cuda
@pytest.mark.parametrize("kind,lora", [("step", False), ("forward_backward", True)])
def test_dp_parameter_gradient_spill_before_accumulation(tmp_path, kind, lora):
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two distinct GPUs")
    from engine_tests.execution._distributed_spill import (
        check_gradient_spill_trace,
        gradient_spill_evidence,
    )
    from shadowspill import Plan

    (tmp_path / "regions.json").write_text(
        json.dumps(
            dict(
                storage="sharded",
                lora=lora,
                retain_sequence=not lora,
                nested=True,
                checkpoint_cuts=False,
                lifetime_mode="replay_compact",
                batch=2,
                sequence=17,
                features=127,
                hidden=259,
                lora_rank=33,
                io_chunk_bytes=[4096, 4092],
                trace_invocations=[0, 1, 2],
                reference_dtype="float64" if kind == "step" else None,
            ),
            indent=2,
        )
        + "\n"
    )
    _verify(tmp_path, kind, "recompute", "regions", memory_policy="gradient_spill")
    paths = {
        path["mode"]: path
        for path in gradient_spill_evidence(Plan.load(tmp_path / "plan.ssp"))
    }
    for invocation in range(3):
        _check_collective_traces(tmp_path, invocation)
        mode = (
            "accumulate" if kind == "forward_backward" and invocation == 1 else "fresh"
        )
        for rank in range(2):
            trace = json.loads(
                (tmp_path / f"rank-{rank}-trace-{invocation}.json").read_text()
            )
            check_gradient_spill_trace(trace, paths[mode])
