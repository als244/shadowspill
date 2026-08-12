"""The neutral core and public vocabulary stay provider-independent."""

from tools.check_naming import main


def test_production_naming_boundaries() -> None:
    assert main() == 0
