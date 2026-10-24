"""Real text windows keep global normalization and storage across candidate sizes."""

import json

import numpy as np
import pytest
import torch

from shadowspill.pytorch.guards import capture_training_signatures
from shadowspill.training._legacy_trainer import Trainer
from shadowspill.training.backends import PyTorch
from shadowspill.training.schedules import WarmupCosine
from workloads.models.llama3.config import Config
from workloads.models.llama3.implementations.pytorch import Model
from workloads.recipes.text.fixed import TokenUpdates
from workloads.recipes.text.objectives import next_token_prediction


def corpus(root, *, tokens=2000, vocabulary=32):
    (np.arange(tokens, dtype=np.uint32) % vocabulary).tofile(root / "train.bin")
    ((np.arange(tokens, dtype=np.uint32) + 7) % vocabulary).tofile(root / "val.bin")
    (root / "meta.json").write_text(
        json.dumps({"vocab_size": vocabulary, "dtype": "uint32"})
    )


def test_all_candidates_read_distinct_rank_windows_with_one_global_normalizer(tmp_path):
    corpus(tmp_path, vocabulary=2048)
    seen = []
    for rank in range(8):
        source = TokenUpdates(
            tmp_path, sequence_length=4, sequences_per_step=8, rank=rank, world_size=8
        )
        update = next(source)
        assert source.next_step == 1
        tokens, targets = update
        assert torch.equal(targets, tokens + 1)
        seen.extend(tokens.flatten().tolist())
        assert len(source.candidates) == 4
        for split in source.candidates.values():
            batches = tuple(split(update))
            signatures = capture_training_signatures(tuple(b for b, _ in batches))
            assert len({s.digest for s in signatures}) == 1
            assert sum(b[0].numel() * scale for b, scale in batches) == 1 / 8
            assert torch.equal(
                torch.cat([b[0] for b, _ in batches], dim=1).flatten(), tokens.flatten()
            )
        state = source.state_dict()
        expected = next(source)
        source.load_state_dict(state)
        assert all(
            torch.equal(a, b) for a, b in zip(next(source), expected, strict=True)
        )
    assert len(seen) == len(set(seen)) == 256


def test_text_example_really_updates_llama_weights_and_matches_unsplit_gradient(
    tmp_path,
):
    import copy

    corpus(tmp_path)
    source = TokenUpdates(tmp_path, sequence_length=8, sequences_per_step=4)
    model = Model(Config(1, 16, 2, 1, 32, 32, max_seq_len=8))
    reference = copy.deepcopy(model)
    update = source.example()
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.01)
    reference.loss(*update).backward()
    optimizer.step()
    before = model.embed.weight.detach().clone()
    with Trainer(
        model,
        objective=next_token_prediction,
        optimizer=torch.optim.SGD,
        optimizer_args={"lr": 0.01},
        microbatches=source.candidates["1_sequences"],
        backend=PyTorch(device="cpu", compile=False),
    ) as trainer:
        trainer.prepare(update)
        result = trainer.step(next(source))
        assert result.step == 1 and result.loss > 0
        assert not torch.equal(trainer.model.embed.weight, before)
        for actual, expected in zip(
            trainer.model.parameters(), reference.parameters(), strict=True
        ):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-7)


def test_candidate_limit_and_too_short_corpus(tmp_path):
    corpus(tmp_path)
    source = TokenUpdates(
        tmp_path,
        sequence_length=4,
        sequences_per_step=64,
        max_sequences_per_microbatch=32,
    )
    assert list(source.candidates) == [
        f"{size}_sequences" for size in (1, 2, 4, 8, 16, 32)
    ]
    with pytest.raises(ValueError, match="shorter"):
        TokenUpdates(tmp_path, sequence_length=4096, sequences_per_step=16)


def test_validation_split_is_distinct_and_cannot_restore_training_progress(tmp_path):
    corpus(tmp_path)
    train = TokenUpdates(tmp_path, sequence_length=8, sequences_per_step=4)
    validation = TokenUpdates(
        tmp_path, split="val", sequence_length=8, sequences_per_step=4
    )
    assert not torch.equal(train.example()[0], validation.example()[0])
    next(train)
    assert validation.next_step == 0
    with pytest.raises(ValueError, match="split"):
        validation.load_state_dict(train.state_dict())


@pytest.mark.parametrize("partial_checkpoint", [False, True])
def test_fixed_updates_evaluate_log_and_resume_with_identical_next_update(
    tmp_path, partial_checkpoint
):
    corpus(tmp_path)
    evaluations = []

    def validation():
        evaluations.append(True)
        return TokenUpdates(
            tmp_path, split="val", sequence_length=8, sequences_per_step=4
        )

    def run(directory, checkpoint=None):
        data = TokenUpdates(tmp_path, sequence_length=8, sequences_per_step=4)
        torch.manual_seed(231)  # The same frozen base on both sides of resume.
        model = Model(Config(1, 16, 2, 1, 32, 32, max_seq_len=8))
        if partial_checkpoint:
            model.requires_grad_(False)
            model.lm_head.requires_grad_(True)
        trainer = Trainer(
            model,
            objective=next_token_prediction,
            optimizer=torch.optim.AdamW,
            schedules={"lr": WarmupCosine(0.01, 0.001, 1, 5)},
            microbatches=data.candidates["1_sequences"],
            backend=PyTorch(device="cpu", compile=False),
            frozen_state_id="fixed-text-base-v1" if partial_checkpoint else None,
        )
        records = []
        trainer.prepare(data.example(), checkpoint=checkpoint)
        trainer.fit(
            data,
            steps=5,
            run_dir=directory,
            logger=records.append,
            eval_data=validation,
            eval_every=2,
            eval_updates=2,
            checkpoint_every=2,
            keep_last=2,
        )
        assert data.next_step == trainer.step_count == 5
        assert records == [
            json.loads(line)
            for line in (directory / "metrics.jsonl").read_text().splitlines()
        ]
        trainer.close()
        state = torch.load(
            directory / "checkpoints" / "step_00000005" / "state.pt",
            weights_only=True,
        )
        return records, state

    whole = tmp_path / "whole"
    records, expected = run(whole)
    if partial_checkpoint:
        assert set(expected["model"]) == {"lm_head.weight"}
    assert [r["step"] for r in records if "eval/loss" in r] == [2, 4, 5]
    assert len(evaluations) == 3
    assert sorted(p.name for p in (whole / "checkpoints").iterdir()) == [
        "step_00000004",
        "step_00000005",
    ]
    resumed, actual = run(tmp_path / "resumed", whole / "checkpoints" / "step_00000004")
    assert len(evaluations) == 4
    for key in ("train/loss", "hyperparameters/lr", "eval/loss"):
        assert [r[key] for r in resumed if key in r] == [
            r[key] for r in records if key in r and r["step"] == 5
        ]
    for name, value in expected["model"].items():
        torch.testing.assert_close(actual["model"][name], value, rtol=0, atol=0)
    for identity, state in expected["optimizer"]["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(
                actual["optimizer"]["state"][identity][key], value, rtol=0, atol=0
            )
