"""GLM-5.3-Flash checkpoint LoRA through independently runnable saved phases."""

from workloads.recipes.phases import run_phases
from workloads.recipes.text.glm53_flash_lora.arguments import parser
from workloads.recipes.text.glm53_flash_lora.settings import configure


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    try:
        root, options = configure(args)
    except ValueError as error:
        cli.error(str(error))
    run_phases(
        "workloads.recipes.text.glm53_flash_lora",
        root,
        args.phase,
        ranks=options["ranks"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
