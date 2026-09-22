"""Which floor a performance cell is judged against."""

from argparse import Namespace

from tools.qualification.performance.verdict import regression_authority
from workloads.full_model import manifests


def _cell(identity: str):
    return next(item for item in manifests() if item.identity == identity)


def test_local_cell_is_judged_against_the_local_floor() -> None:
    cell = _cell("mlops_llama3")
    assert regression_authority(cell, Namespace()) == cell.regression_tokens_per_second


def test_remote_cell_is_judged_against_the_remote_floor() -> None:
    cell = _cell("mlops_llama3")
    arguments = Namespace(remote_spill="192.0.2.1:17800:1024")
    assert (
        regression_authority(cell, arguments)
        == cell.remote_regression_tokens_per_second
    )


def test_cell_without_authorities_is_judged_against_nothing() -> None:
    cell = _cell("pytorch_llama3")
    assert regression_authority(cell, Namespace()) is None
    assert regression_authority(cell, Namespace(remote_spill="h:1:1")) is None
