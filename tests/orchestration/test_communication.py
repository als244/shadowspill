"""Candidate composition retains exact probe geometry and independent profiles."""

import json
from dataclasses import replace

import pytest

from orchestration.communication import communication_probes
from shadowspill import (
    CommunicationInfo,
    Parallelism,
    ProfiledProgram,
    Profiles,
    apply_profiles,
)
from tests.shadowspill.offline._example import profiles
from tests.shadowspill.offline.test_sharded_forward import forward


def reordered(source):
    shard = source.bindings["weight_shards"]["weight_storage"]
    return replace(
        source,
        bindings={
            **source.bindings,
            "weight_shards": {
                "weight_storage": {**shard, "groups": ((3, 1, 0, 2),)},
            },
        },
    )


def test_union_preserves_rank_order_and_deduplicates_across_candidates():
    source = forward()
    first, requests = communication_probes(iter((source, source, reordered(source))))
    assert first is source
    assert requests == [
        {
            "operation": "all_gather",
            "ranks": ranks,
            "tensor_bytes": 64,
            "dtype": "torch.uint8",
            "inplace": True,
        }
        for ranks in ((0, 1, 2, 3), (3, 1, 0, 2))
    ]


@pytest.mark.parametrize(
    "change,message",
    [
        ({"parallelism": Parallelism.dp(ranks_per_replica=2)}, "same parallelism"),
        ({"setup": {"gpu": "different"}}, "same build setup"),
    ],
)
def test_mixed_calibration_contexts_are_rejected(change, message):
    source = forward()
    with pytest.raises(ValueError, match=message):
        communication_probes((source, replace(source, **change)))


def test_empty_calibration_is_explicit():
    with pytest.raises(ValueError, match="at least one Program"):
        communication_probes(())


def test_recipe_shares_probes_but_profiles_every_train_and_eval_candidate(
    tmp_path, monkeypatch
):
    from workloads.recipes.training import communicate as phase

    source = forward()
    options = {
        "candidates": [{"path": "a"}, {"path": "b"}],
        "eval_every": 1,
        "ranks": 4,
        "warmup_seconds": 0.01,
        "measure_seconds": 0.02,
    }
    paths = ("a/training.ssp", "a/evaluation.ssp", "b/training.ssp", "b/evaluation.ssp")
    expected_profiles = {}
    for index, relative in enumerate(paths):
        program = replace(source if index < 2 else reordered(source), name=relative)
        program.save(tmp_path / "programs" / relative)
        profile = profiles()
        profile.setup = dict(program.setup)
        profiled = apply_profiles(program, profile)
        expected_profiles[relative] = profiled.profiles.tasks
        profiled.save(tmp_path / "profiled" / relative)
    monkeypatch.setattr(phase, "settings", lambda _: (tmp_path, options))
    monkeypatch.setattr(phase.dist, "init_process_group", lambda *a, **k: None)
    monkeypatch.setattr(phase.dist, "destroy_process_group", lambda: None)
    monkeypatch.setattr(phase.dist, "get_rank", lambda: 0)
    calls, measured = [], []
    info = CommunicationInfo(64e6, 64e6)

    def communication(**kwargs):
        calls.append(kwargs)
        return info

    def overlap(profiled, communication, **kwargs):
        assert communication is info
        measured.append(profiled.program.name)
        return Profiles(setup=dict(profiled.program.setup))

    monkeypatch.setattr(phase, "measure_communication", communication)
    monkeypatch.setattr(phase, "measure_overlap", overlap)
    phase.run(tmp_path)
    assert len(calls) == 1
    assert len(calls[0]["collectives"]) == 2
    assert calls[0]["rank_map"] == {rank: rank for rank in range(4)}
    assert measured == list(paths)
    for relative in paths:
        assert CommunicationInfo.load(tmp_path / "communication" / relative) == info
        assert (
            ProfiledProgram.load(tmp_path / "profiled" / relative).profiles.tasks
            == expected_profiles[relative]
        )
    events = [
        json.loads(line)
        for line in (tmp_path / "logs/communication-rank-00000.jsonl")
        .read_text()
        .splitlines()
    ]
    assert (
        sum(
            e["name"] == "communication.measure" and e["boundary"] == "end"
            for e in events
        )
        == 1
    )
    assert (
        sum(e["name"] == "overlap.measure" and e["boundary"] == "end" for e in events)
        == 4
    )
