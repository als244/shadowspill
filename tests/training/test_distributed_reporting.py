"""Real CPU processes check nonblocking observations and grouped offline W&B."""

from __future__ import annotations

import json
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.distributed_c10d import _get_process_group_store

from shadowspill.training._reporting import sum_rank_records
from shadowspill.training.logging import DistributedLogger


def reporting_worker(rank, root, use_wandb, mismatched):
    root = Path(root)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(root / "store"),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        options = (
            dict(project="shadowspill-distributed-test", mode="offline", group="test")
            if use_wandb
            else None
        )
        store = _get_process_group_store(dist.group.WORLD)
        logger = DistributedLogger(
            dist.group.WORLD,
            run_dir=root,
            wandb=options,
            max_pending=4,
            timeout=10,
        )
        try:
            # Rank 1 waits until rank 0's submit returns. A hidden reporting
            # collective/barrier would deadlock here instead of reaching ready.
            if rank == 1:
                store.wait(["rank0_submitted"], timedelta(seconds=10))
            for step in range(1, 9):
                logger(
                    {
                        "step": step + int(mismatched and rank == 1),
                        "train/loss": float(rank + 1),
                        "train/step_seconds": (rank + 1) * 0.2,
                        "train/elapsed_seconds": step * (rank + 1) * 0.2,
                        "train/work_units": 3 + rank * 2,
                        "train/private_metric": rank + step,
                    }
                )
                if rank == 0 and step == 1:
                    store.set("rank0_submitted", "1")
                # Reuse the bounded mailbox slots, allowing the CPU worker time
                # to consume each record. This wait is test pacing only.
                time.sleep(0.03)
            if use_wandb:
                assert logger.local.run.step == 8
            logger.close()
        except RuntimeError as error:
            if not mismatched:
                raise
            (root / f"failure-{rank}.txt").write_text(str(error))
        finally:
            try:
                logger.close(exit_code=int(mismatched))
            except RuntimeError:
                if not mismatched:
                    raise
        if use_wandb:
            assert logger.local.run.group == "test"
            assert logger.local.run.job_type == "rank"
            if rank == 0:
                assert logger.aggregate.run.group == "test"
                assert logger.aggregate.run.job_type == "aggregate"
        if not mismatched:
            (root / f"complete-{rank}").write_text("ok")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("use_wandb", [False, True])
def test_distributed_records_and_wandb_keep_rank_and_aggregate_runs(
    tmp_path, monkeypatch, use_wandb
):
    if use_wandb:
        pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_SILENT", "true")
    mp.spawn(reporting_worker, args=(tmp_path, use_wandb, False), nprocs=2, join=True)
    records = [
        json.loads(line)
        for line in (tmp_path / "aggregate/metrics.jsonl").read_text().splitlines()
    ]
    assert [r["step"] for r in records] == list(range(1, 9))
    for record in records:
        assert record["train/loss"] == 3
        assert record["train/step_seconds"] == pytest.approx(0.4)
        assert record["train/work_units"] == 8
        assert record["train/units_per_second"] == pytest.approx(20)
        assert "train/private_metric" not in record
    assert (tmp_path / "complete-0").is_file()
    assert (tmp_path / "complete-1").is_file()
    if use_wandb:
        assert len(list(tmp_path.glob("**/run-*.wandb"))) == 3


def test_reporting_order_mismatch_fails_both_ranks_without_hanging(tmp_path):
    mp.spawn(reporting_worker, args=(tmp_path, False, True), nprocs=2, join=True)
    for rank in range(2):
        assert (
            "step/phase order differs" in (tmp_path / f"failure-{rank}.txt").read_text()
        )


def test_default_aggregation_does_not_guess_eval_weighting():
    result = sum_rank_records(
        [
            {"step": 2, "eval/loss": 1, "eval/seconds": 0.2},
            {"step": 2, "eval/loss": 3, "eval/seconds": 0.4},
        ]
    )
    assert result == {"step": 2, "eval/seconds": 0.4}
