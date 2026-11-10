"""Optional CLI spelling; the model-independent runner owns all phases."""

from workloads.recipes.text.qwen35_lora.api import run
from workloads.recipes.text.qwen35_lora.arguments import parser
from workloads.recipes.training.cli import arguments


def main(argv=None):
    command = parser()
    phase, config = arguments(command, argv)
    try:
        run(config, phase=phase)
    except ValueError as error:
        command.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
