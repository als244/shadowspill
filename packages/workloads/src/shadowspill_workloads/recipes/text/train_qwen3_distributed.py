"""Optional CLI; public Python scripts use Model.from_config and train."""

from shadowspill.training.preparation import run
from shadowspill.training.preparation.cli import arguments
from shadowspill_workloads.recipes.text.qwen3.arguments import parser


def main(argv=None):
    command = parser()
    phase, config = arguments(command, argv)
    try:
        run("shadowspill_workloads.recipes.text.qwen3.adapter", config, phase=phase)
    except ValueError as error:
        command.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
