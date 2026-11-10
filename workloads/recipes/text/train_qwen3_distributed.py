"""Optional CLI; public Python scripts use Model.from_config and train."""

from workloads.recipes.text.qwen3.arguments import parser
from workloads.recipes.training import run
from workloads.recipes.training.cli import arguments


def main(argv=None):
    command = parser()
    phase, config = arguments(command, argv)
    try:
        run("workloads.recipes.text.qwen3.adapter", config, phase=phase)
    except ValueError as error:
        command.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
