"""The text helper aligns only its explicitly declared original-token fields."""

import numpy as np
import pytest

from shadowspill_client import Datum
from shadowspill_client.text import text_batch, text_datum


def test_text_shift_preserves_target_mask_and_signed_weights():
    datum = text_datum(
        [11, 12, 13, 14],
        loss_mask=[False, False, True, True],
        weights=[99, 1, -2, 3],
    )
    np.testing.assert_array_equal(datum.inputs["tokens"], [11, 12, 13])
    np.testing.assert_array_equal(datum.loss_inputs["target_tokens"], [12, 13, 14])
    np.testing.assert_array_equal(datum.loss_inputs["loss_mask"], [False, True, True])
    np.testing.assert_array_equal(datum.loss_inputs["weights"], [1, -2, 3])
    assert datum.loss_inputs["weights"].dtype == np.float32


def test_default_fields_remain_schema_defaults_and_explicit_datum_is_unshifted():
    tokens = np.array([1, 2, 3], dtype=np.int64)
    datum = text_datum(tokens)
    assert set(datum.loss_inputs) == {"target_tokens"}
    assert np.shares_memory(datum.inputs["tokens"], tokens)
    assert Datum(inputs={"tokens": tokens}).inputs["tokens"] is tokens
    weighted = text_datum(tokens, weights=-0.5)
    np.testing.assert_array_equal(weighted.loss_inputs["weights"], [-0.5, -0.5])


@pytest.mark.parametrize(
    "field,values",
    [
        ("microbatch_tokens", []),
        ("microbatch_tokens", [0]),
        ("max_sequences_per_microbatch", 0),
        ("max_sequences_per_microbatch", True),
        ("projection_chunks", []),
        ("projection_chunks", [-1]),
        ("moe_chunks", [True]),
        ("moe_chunks", [0, 0]),
        ("retention", []),
        ("retention", ["cost-input", "cost-input"]),
        ("retention", ["unknown"]),
    ],
)
def test_invalid_build_candidates_fail_on_the_client(field, values):
    from shadowspill_client.text import build_candidates

    options = {"microbatch_tokens": [8192], field: values}
    with pytest.raises(ValueError):
        build_candidates(**options)


def test_per_microbatch_sequence_capacity_survives_portable_roundtrip():
    from shadowspill_client import BuildCandidates
    from shadowspill_client.text import build_candidates

    candidates = build_candidates(
        microbatch_tokens=[32768, 65536], max_sequences_per_microbatch=128
    )
    restored = BuildCandidates.from_dict(candidates.to_dict())
    assert restored == candidates
    assert [case.max_datums for case in restored.cases] == [128, 128]


@pytest.mark.parametrize(
    "tokens,options",
    [
        ([], {}),
        ([1], {}),
        ([True, False], {}),
        ([1.5, 2.5], {}),
        ([1, -2], {}),
        ([[1, 2]], {}),
        (np.array([1, 2**63], dtype=np.uint64), {}),
        ([1, 2, 3], {"loss_mask": [True, False]}),
        ([1, 2, 3], {"loss_mask": [0, 2, 0]}),
        ([1, 2, 3], {"weights": [1, 2]}),
        ([1, 2, 3], {"weights": float("nan")}),
    ],
)
def test_malformed_text_fields_are_refused(tokens, options):
    with pytest.raises(ValueError):
        text_datum(tokens, **options)


def test_batch_mean_counts_unmasked_targets_across_unequal_sequences():
    batch = text_batch(
        [[1, 2, 3, 4], [5, 6], [7, 8, 9]],
        loss_masks=[[True, True, False, True], [False, True], [True, False, False]],
    )
    # Three trained targets across all records, not an average of record means.
    losses = [np.array([2.0, 100.0, 4.0]), np.array([6.0]), np.array([90.0, 90.0])]

    def loss(records, values):
        return sum(
            np.sum(value * d.loss_inputs["weights"] * d.loss_inputs["loss_mask"])
            for d, value in zip(records, values, strict=True)
        )

    expected = np.mean([2.0, 4.0, 6.0])
    assert loss(batch, losses) == pytest.approx(expected)
    # Worker partitions or split backward calls keep the original denominator.
    assert loss(batch[:1], losses[:1]) + loss(batch[1:], losses[1:]) == pytest.approx(
        expected
    )
    assert all(np.allclose(d.loss_inputs["weights"], 1 / 3) for d in batch)


def test_batch_mean_without_masks_uses_target_count():
    batch = text_batch(iter([[1, 2], [3, 4, 5, 6]]))
    assert all("loss_mask" not in d.loss_inputs for d in batch)
    assert all(np.allclose(d.loss_inputs["weights"], 1 / 4) for d in batch)


@pytest.mark.parametrize(
    "sequences,masks",
    [([], None), ([[1, 2]], [[True, False]]), ([[1, 2]], []), ([], [[True, True]])],
)
def test_empty_supervision_and_misaligned_batch_masks_are_refused(sequences, masks):
    with pytest.raises(ValueError):
        text_batch(sequences, loss_masks=masks)
