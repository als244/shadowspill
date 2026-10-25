"""GLM next-token geometry, rank ownership and exact tokenizer identity."""

import json

import numpy as np
import pytest
import torch

from workloads.recipes.text.glm import GLMTokenUpdates, fixed_batch_learning_report


def test_glm_updates_preserve_shift_rank_ownership_and_resume(tmp_path):
    values = np.arange(1041, dtype=np.uint32)
    values.tofile(tmp_path / "train.bin")
    (tmp_path / "meta.json").write_text(
        json.dumps({"vocab_size": 2048, "dtype": "uint32"})
    )
    ranks = [
        GLMTokenUpdates(
            tmp_path, sequence_length=65, sequences_per_step=2, rank=rank, world_size=2
        )
        for rank in range(2)
    ]
    for rank, data in enumerate(ranks):
        batch, scale = next(data.microbatches(next(data), sequences_per_microbatch=2))
        tokens, targets, boundaries, cumulative, chunks = batch
        assert tokens[0].item() == rank * 130
        torch.testing.assert_close(targets, tokens + 1)
        assert boundaries == (0, 65, 130)
        assert cumulative.tolist() == list(boundaries)
        assert chunks.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
        assert scale == 1 / 260
        state = data.state_dict()
        expected = next(data)
        data.load_state_dict(state)
        actual = next(data)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b)
    assert ranks[0].example()[0][-1, -1] < ranks[1].example()[0][0, 0]


def test_fixed_batch_keeps_rank_ownership_and_checkpoint_policy(tmp_path):
    np.arange(4097, dtype=np.uint32).tofile(tmp_path / "train.bin")
    (tmp_path / "meta.json").write_text(
        json.dumps({"vocab_size": 8192, "dtype": "uint32"})
    )
    geometry = dict(sequence_length=128, sequences_per_step=1, rank=1, world_size=2)
    data = GLMTokenUpdates(tmp_path, repeat_first_update=True, **geometry)
    first = next(data)
    assert first[0][0, 0] == 128
    for _ in range(3):
        repeated = next(data)
        for actual, expected in zip(repeated, first, strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    state = data.state_dict()
    assert state["next_step"] == 4
    restored = GLMTokenUpdates(tmp_path, repeat_first_update=True, **geometry)
    restored.load_state_dict(state)
    assert restored.next_step == 4
    for actual, expected in zip(next(restored), first, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    rolling = GLMTokenUpdates(tmp_path, **geometry)
    with pytest.raises(ValueError, match="fixed-batch policy"):
        rolling.load_state_dict(state)


def test_fixed_batch_learning_gate_rejects_flat_rising_and_nonfinite_losses():
    good = fixed_batch_learning_report([3.0, 2.9, 2.8, 2.7], minimum_relative_drop=0.01)
    assert good["passed"]
    assert good["relative_drop"] == pytest.approx(1 / 15)
    for losses in ([3.0] * 4, [3.0, 3.1, 3.2, 3.3], [3.0, 2.999, 2.999, 2.999]):
        assert not fixed_batch_learning_report(losses, minimum_relative_drop=0.01)[
            "passed"
        ]
    with pytest.raises(ValueError, match="finite"):
        fixed_batch_learning_report(
            [3.0, 2.9, float("nan"), 2.7], minimum_relative_drop=0.01
        )


def test_prepared_corpus_records_checkpoint_tokenizer_identity(tmp_path):
    from workloads.recipes.text.data.prepare import write_meta
    from workloads.recipes.text.data.tokenizer import tokenizer_identity

    class Tokenizer:
        eos_token_id = 3

        def __len__(self):
            return 4

    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "tokenizer.json").write_text('{"vocab": ["a", "b", "c"]}')
    config = checkpoint / "tokenizer_config.json"
    config.write_text('{"eos_token": "c"}')
    counts = dict(files=["text.parquet"], train_tokens=256, val_tokens=128)
    meta = write_meta(tmp_path, Tokenizer(), str(checkpoint), counts)
    assert meta["tokenizer_sha256"] == tokenizer_identity(checkpoint)
    assert json.loads((tmp_path / "meta.json").read_text()) == meta
    # A changed special-token policy must fail the recipe's exact identity check.
    config.write_text('{"eos_token": "b"}')
    assert meta["tokenizer_sha256"] != tokenizer_identity(checkpoint)


@pytest.mark.parametrize("visible,selected", [("GPU-a,GPU-b", "GPU-b"), (None, "1")])
def test_recipe_worker_keeps_global_rank_and_maps_assigned_device(
    monkeypatch, visible, selected
):
    import os
    import sys

    from workloads.recipes import logical_device_worker as worker

    monkeypatch.setenv("LOCAL_RANK", "1")
    monkeypatch.setenv("RANK", "5")
    monkeypatch.setenv("WORLD_SIZE", "8")
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(sys, "argv", ["worker", "target.module", "--steps", "12"])
    calls = []
    monkeypatch.setattr(
        worker.runpy, "run_module", lambda *a, **k: calls.append((a, k))
    )
    worker.main()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == selected
    assert os.environ["LOCAL_RANK"] == "0"
    assert (os.environ["RANK"], os.environ["WORLD_SIZE"]) == ("5", "8")
    assert sys.argv == ["target.module", "--steps", "12"]
    assert calls == [(("target.module",), dict(run_name="__main__", alter_sys=True))]
