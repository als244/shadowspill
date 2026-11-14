"""The text helper aligns only its explicitly declared original-token fields."""

import numpy as np
import pytest

from shadowspill_client import Datum
from shadowspill_client.text import text_datum


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
