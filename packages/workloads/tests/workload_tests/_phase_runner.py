"""Isolated phase-fixture launcher used only by workload integration tests."""

from shadowspill.orchestration.launch.processes import worker_command
from shadowspill.training.preparation.configuration import configure
from shadowspill.training.preparation.phases import run_phases


def run(recipe, values, *, phase):
    root, options = configure(recipe, values, phase)
    if phase == "execute":
        import subprocess

        subprocess.run(
            worker_command(
                "workload_tests._phase_execute", [root], ranks=options["ranks"]
            ),
            check=True,
        )
    else:
        run_phases(
            "shadowspill.training.preparation",
            root,
            phase,
            ranks=options["ranks"],
            preparation_ranks=options["preparation_ranks"],
            phases=("build", "profile", "communicate", "plan")
            if options["ranks"] > 1
            else ("build", "profile", "plan"),
        )
