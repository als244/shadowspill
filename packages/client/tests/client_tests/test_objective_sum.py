import subprocess
import sys

import pytest

from shadowspill_client import Objective


@pytest.mark.parametrize(
    "terms,weights",
    [
        ({}, None),
        ({"bad.name": Objective("external/loss@1")}, None),
        ({"loss": Objective("external/loss@1")}, {"other": 1.0}),
        ({"loss": Objective("external/loss@1")}, {"loss": float("nan")}),
        ({"nested": Objective("shadowspill/sum@1")}, None),
    ],
)
def test_invalid_sum_settings(terms, weights):
    with pytest.raises(ValueError):
        Objective.sum(terms, weights=weights)


def test_sum_roundtrip_remains_framework_and_engine_free():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc, json, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'shadowspill', 'shadowspill_workloads'}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from shadowspill_client import Objective
value = Objective.sum({
    'main': Objective('external/loss@1'),
    'balance': Objective('external/balance@1'),
}, weights={'balance': 0.001})
assert Objective.from_dict(json.loads(json.dumps(value.to_dict()))) == value
assert value.defaults == {'balance.weight': 0.001}
""",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
