"""The suite chooses supported storage before fresh-process tests import it."""

import pytest

from tests import precision


@pytest.mark.parametrize(
    "supports_bf16, expected", [(True, "bfloat16"), (False, "float16")]
)
def test_automatic_suite_dtype(monkeypatch, supports_bf16, expected):
    monkeypatch.delenv("SHADOWSPILL_TEST_DTYPE", raising=False)
    probes = []

    def probe(*, allow_cpu):
        probes.append(allow_cpu)
        return {"device": "test", "bf16": supports_bf16}

    monkeypatch.setattr(precision, "detect_device_precision", probe)
    assert precision.select_test_dtype() == expected
    assert probes == [True]


def test_suite_cli_override_wins_without_a_probe(monkeypatch):
    monkeypatch.setenv("SHADOWSPILL_TEST_DTYPE", "bfloat16")
    monkeypatch.setattr(
        precision,
        "detect_device_precision",
        lambda **kwargs: pytest.fail("explicit selection must not probe"),
    )
    assert precision.select_test_dtype("float16") == "float16"


def test_suite_children_inherit_dtype_without_a_probe(monkeypatch):
    monkeypatch.setenv("SHADOWSPILL_TEST_DTYPE", "float16")
    monkeypatch.setattr(
        precision,
        "detect_device_precision",
        lambda **kwargs: pytest.fail("child must inherit the parent's selection"),
    )
    assert precision.select_test_dtype() == "float16"
    assert str(precision.low_precision_dtype()) == "torch.float16"


def test_invalid_inherited_suite_dtype_is_rejected(monkeypatch):
    monkeypatch.setenv("SHADOWSPILL_TEST_DTYPE", "invalid")
    with pytest.raises(ValueError, match="float16 or bfloat16"):
        precision.select_test_dtype()
