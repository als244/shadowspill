"""Storage selection must leave the numerical authority and cases unchanged."""

from dataclasses import replace

import pytest

from qualification.numerical.matrix import _case_commands, _CaseOptions, _parser
from qualification.numerical.spill import configured_spill, spill_arguments
from qualification.numerical.tolerances import SPILL_BUDGET
from shadowspill.ssd import SSDPool


def test_ssd_changes_only_the_planned_arm(tmp_path):
    parser = _parser()
    arguments = parser.parse_args(
        [
            "--spill-pool",
            "ssd",
            "--ssd-directory",
            str(tmp_path),
            "--ssd-staging-mib",
            "128",
            "--ssd-chunk-mib",
            "4",
            "--ssd-queue-depth",
            "8",
        ]
    )
    pool = configured_spill(parser, arguments)
    assert isinstance(pool, SSDPool)
    assert pool.capacity == SPILL_BUDGET
    assert configured_spill(parser, parser.parse_args(spill_arguments(pool))) == pool
    options = _CaseOptions(
        environment={},
        reference_directory=tmp_path,
        regenerate_reference=True,
        seed=123,
        model_config="{}",
        data_geometry=None,
        case_factory=None,
        case_options=[],
        optimizer_ordering="stage_interleaved",
        data_ordering=None,
        empty_caches=False,
        cache_directory=None,
        detailed_artifacts=False,
    )

    def commands(option):
        return _case_commands(
            "llama3",
            "pytorch",
            6 << 30,
            tmp_path / "ref.pt",
            tmp_path / "result.json",
            option,
        )

    host = commands(options)
    on_ssd = commands(replace(options, spill_pool=pool))
    assert on_ssd[0] == host[0]
    assert on_ssd[1] == host[1] + spill_arguments(pool)


def test_ssd_requires_explicit_storage_and_checks_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv("SHADOWSPILL_SSD_DIRECTORY", raising=False)
    parser = _parser()
    assert configured_spill(parser, parser.parse_args([])) is None
    with pytest.raises(SystemExit):
        configured_spill(parser, parser.parse_args(["--spill-pool", "ssd"]))
    for extra in (
        ["--ssd-staging-mib", "0"],
        ["--ssd-chunk-mib", "-1"],
        ["--ssd-queue-depth", "0"],
        ["--remote-spill", "localhost:1234:1024"],
    ):
        with pytest.raises(SystemExit):
            configured_spill(
                parser,
                parser.parse_args(
                    [
                        "--spill-pool",
                        "ssd",
                        "--ssd-directory",
                        str(tmp_path),
                        *extra,
                    ]
                ),
            )
    monkeypatch.setenv("SHADOWSPILL_SSD_DIRECTORY", str(tmp_path))
    parser = _parser()
    assert isinstance(
        configured_spill(parser, parser.parse_args(["--spill-pool", "ssd"])),
        SSDPool,
    )
