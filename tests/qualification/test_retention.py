"""A cell's execution-only files go after its verdict unless retention is asked for."""

from __future__ import annotations

from pathlib import Path

from qualification.numerical.matrix.commands import CaseOptions, case_commands
from qualification.performance.manifest import _manifest_with_overrides
from qualification.performance.matrix.arguments import create_parser
from qualification.performance.matrix.command import cell_command
from qualification.pipeline.retention import EXECUTION_FILES, remove_execution_files


def test_only_the_execution_files_leave_the_archive(tmp_path: Path) -> None:
    kept = ("program.ssp", "profiles.ssp", "plan.ssp", "settings.json", "phases.json")
    for name in (*kept, *EXECUTION_FILES):
        (tmp_path / name).write_bytes(b"x")
    remove_execution_files(tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(kept)
    # A second removal, or one before execute ever wrote them, is not an error.
    remove_execution_files(tmp_path)


def _numerical_options(tmp_path: Path, **overrides: object) -> CaseOptions:
    return CaseOptions(
        environment={},
        reference_directory=tmp_path,
        regenerate_reference=False,
        seed=1,
        model_config="{}",
        data_geometry=None,
        case_factory=None,
        case_options=[],
        optimizer_ordering="stage_interleaved",
        data_ordering=None,
        empty_caches=False,
        cache_directory=None,
        detailed_artifacts=False,
        **overrides,  # type: ignore[arg-type]
    )


def test_numerical_cells_retain_execution_files_only_when_asked(
    tmp_path: Path,
) -> None:
    (tmp_path / "ref.pt").write_bytes(b"")

    def planned(options: CaseOptions) -> list[str]:
        commands = case_commands(
            "llama3",
            "pytorch",
            6 << 30,
            tmp_path / "ref.pt",
            tmp_path / "r.json",
            options,
        )
        # The warm-up and reference arms execute nothing worth retaining.
        assert not any("--retain-execution-files" in c for c in commands[:-1])
        return commands[-1]

    assert "--retain-execution-files" not in planned(_numerical_options(tmp_path))
    assert "--retain-execution-files" in planned(
        _numerical_options(tmp_path, retain_execution_files=True)
    )


def test_performance_cells_retain_execution_files_only_when_asked() -> None:
    manifest = _manifest_with_overrides("llama3", "mlops", spill_budget_gib=None)

    def command(*flags: str) -> list[str]:
        return cell_command(
            manifest, create_parser().parse_args(list(flags)), Path("out/c.json"), {}
        )

    assert "--retain-execution-files" not in command()
    assert "--retain-execution-files" in command("--retain-execution-files")
