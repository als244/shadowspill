"""Optional CLI spelling; the model-independent runner owns all phases."""

from shadowspill.training.preparation.cli import arguments
from shadowspill_workloads.recipes.text.glm53_flash_lora.api import run
from shadowspill_workloads.recipes.text.glm53_flash_lora.arguments import parser


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
