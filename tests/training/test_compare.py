"""training.compare reads one process's run and a multi-rank run alike."""

import json

from training.compare import packing, read


def _write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


LOSSES = [{"step": 1, "train/loss": 2.0}, {"step": 1, "eval/loss": 1.5}]


def test_ranks_and_microbatches_dividing_a_step_differently_compare(tmp_path):
    single = tmp_path / "single"
    _write(single / "metrics.jsonl", LOSSES)
    _write(
        single / "packing.jsonl",
        [{"step": 1, "microbatches": [[[3, 10], [1, 20]], [[2, 5]]]}],
    )
    # A heterogeneous run: each node keeps its own directory, the aggregate
    # beside the first rank.
    nodes = tmp_path / "nodes_run"
    _write(nodes / "nodes/00/aggregate/metrics.jsonl", LOSSES)
    _write(
        nodes / "nodes/00/rank-00000/packing.jsonl",
        [{"step": 1, "microbatches": [[[2, 5], [3, 10]]]}],
    )
    _write(
        nodes / "nodes/01/rank-00001/packing.jsonl",
        [{"step": 1, "microbatches": [[[1, 20]]]}],
    )
    # A multi-rank run on one machine.
    ranks = tmp_path / "ranks_run"
    _write(ranks / "aggregate/metrics.jsonl", LOSSES)
    _write(
        ranks / "rank-00000/packing.jsonl", [{"step": 1, "microbatches": [[[1, 20]]]}]
    )
    _write(
        ranks / "rank-00001/packing.jsonl",
        [{"step": 1, "microbatches": [[[3, 10]], [[2, 5]]]}],
    )
    for run in (nodes, ranks):
        assert read(run) == read(single)
        assert packing(run) == packing(single)


def test_a_resumed_rank_replaces_its_step_and_real_differences_show(tmp_path):
    run = tmp_path / "run"
    _write(run / "aggregate/metrics.jsonl", LOSSES)
    _write(
        run / "rank-00000/packing.jsonl",
        [
            {"step": 1, "microbatches": [[[9, 99]]]},
            {"step": 1, "microbatches": [[[1, 20]]]},
        ],
    )
    _write(run / "rank-00001/packing.jsonl", [{"step": 1, "microbatches": [[[2, 5]]]}])
    assert packing(run) == {1: ["[1, 20]", "[2, 5]"]}
    other = tmp_path / "other"
    _write(other / "metrics.jsonl", LOSSES)
    _write(other / "packing.jsonl", [{"step": 1, "microbatches": [[[1, 20], [2, 6]]]}])
    assert packing(other) != packing(run)
