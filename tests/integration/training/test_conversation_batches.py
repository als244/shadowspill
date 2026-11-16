"""Variable conversation boundaries and global assistant-token normalization."""

import json

import numpy as np
import pytest

from examples.training.text_data import ConversationBatches


def corpus(path, rows):
    tokens, masks = [], []
    for sequence, mask in rows:
        tokens.extend([*sequence, 99])
        masks.extend([*mask, 0])
    np.asarray(tokens, dtype=np.uint32).tofile(path / "train.bin")
    np.asarray(masks, dtype=np.uint8).tofile(path / "train.mask.bin")
    (path / "meta.json").write_text(
        json.dumps(
            dict(
                dtype="uint32",
                vocab_size=100,
                rendered="chat_template",
                mask="assistant",
                eos_id=99,
            )
        )
    )
    return path


def read(path, **kwargs):
    return ConversationBatches(path, max_sequence_length=8, tokens_per_step=8, **kwargs)


def test_complete_variable_conversations_pack_by_tokens_and_shift_masks(tmp_path):
    source = read(
        corpus(
            tmp_path,
            [
                ([1, 2, 3, 4], [0, 0, 1, 1]),
                ([5, 6, 7, 8, 9, 10], [0, 0, 0, 1, 1, 1]),
                ([11, 12, 13], [0, 0, 1]),
            ],
        )
    )
    # Exercise a conversation crossing scanner chunks without whole-file indexing.
    source.SCAN_TOKENS = 4
    batches = list(source)
    assert [[len(d.inputs["tokens"]) for d in b] for b in batches] == [[3, 5], [2]]
    assert [d.inputs["tokens"].tolist() for d in batches[0]] == [
        [1, 2, 3],
        [5, 6, 7, 8, 9],
    ]
    assert [d.loss_inputs["target_tokens"].tolist() for d in batches[0]] == [
        [2, 3, 4],
        [6, 7, 8, 9, 10],
    ]
    assert [d.loss_inputs["loss_mask"].tolist() for d in batches[0]] == [
        [False, True, True],
        [False, False, True, True, True],
    ]
    assert all(np.allclose(d.loss_inputs["weights"], 1 / 5) for d in batches[0])
    for batch in batches:
        assert sum(
            float((d.loss_inputs["weights"] * d.loss_inputs["loss_mask"]).sum())
            for d in batch
        ) == pytest.approx(1)
    batches[0][0].inputs["tokens"][0] = 98
    assert next(iter(source))[0].inputs["tokens"][0] == 1


def test_count_limit_and_overlong_unsupervised_skips_keep_conversations_intact(
    tmp_path,
):
    source = read(
        corpus(
            tmp_path,
            [
                (list(range(12)), [1] * 12),  # Too long; never truncate.
                (
                    [2, 3],
                    [1, 0],
                ),  # Only original first token marked; no trained target.
                ([4], [1]),
                ([5, 6, 7], [0, 1, 1]),
                ([8, 9], [0, 1]),
            ],
        ),
        max_sequences=1,
    )
    batches = list(source)
    assert len(batches) == 2
    assert [b[0].inputs["tokens"].tolist() for b in batches] == [[5, 6], [8]]


def test_no_eligible_conversations_is_explicit(tmp_path):
    source = read(corpus(tmp_path, [([1, 2, 3], [0, 0, 0])]))
    with pytest.raises(ValueError, match="no supervised conversations"):
        list(source)


@pytest.mark.parametrize("damage", ["mask", "metadata", "boundary"])
def test_missing_chat_contract_is_rejected(tmp_path, damage):
    corpus(tmp_path, [([1, 2, 3], [0, 0, 1])])
    if damage == "mask":
        (tmp_path / "train.mask.bin").unlink()
    elif damage == "metadata":
        (tmp_path / "meta.json").write_text('{"vocab_size": 100}')
    else:
        np.asarray([1, 2, 3, 4], dtype=np.uint32).tofile(tmp_path / "train.bin")
    with pytest.raises(ValueError, match=r"missing loss mask|prepare_chat|incomplete"):
        read(tmp_path)


@pytest.mark.parametrize(
    "setting",
    [
        {"tokens_per_step": 0},
        {"max_sequence_length": 9},
        {"max_sequences": True},
    ],
)
def test_invalid_limits_fail_before_opening_corpus(tmp_path, setting):
    settings = dict(max_sequence_length=8, tokens_per_step=8, max_sequences=2)
    settings.update(setting)
    with pytest.raises(ValueError, match=r"positive integers|exceeds"):
        ConversationBatches(tmp_path, **settings)
