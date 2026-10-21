"""Launch one case of any harness.

    python -m baselines.common.launch --harness NAME --gpus N --output DIR \\
        [--python PATH] <the harness's settings flags>

Every harness starts here, from the sweep or by hand. The settings are
validated in this interpreter, the one that runs the sweep, which can
import the catalog with its implementations, readers and LoRA; the case is
then resolved: the row's description, the model's catalog size, the useful
work of a step counted from the catalog model built on ``meta``, a summary
of what was built, whatever the harness's ``Settings.resolve`` adds (a
recipe harness's Hugging Face configuration), and the environment variables
the case's settings decide (``Settings.environment``). The case is written to
``DIR/case.json`` and torchrun replaces this process, running the harness's
trainer in its own interpreter with ``--output DIR --case DIR/case.json``,
in this process's environment with the case's variables set or unset.
That interpreter is ``--python``, or the environment the harness names
(``ENVIRONMENT``) beside the running one.
"""

from __future__ import annotations

import argparse
import os
import sys
from importlib import import_module
from pathlib import Path

from baselines.common.results import write_json

#: What a harness's settings module provides.
REQUIRED = (
    "Settings",
    "SOURCE_PATHS",
    "ENVIRONMENT",
    "add_arguments",
    "from_namespace",
)


def harness_module(name):
    """The settings module of a harness, checked for what is called."""

    try:
        module = import_module(f"baselines.{name}.settings")
    except ModuleNotFoundError as error:
        raise ValueError(f"no harness {name!r}: {error}") from error
    missing = [attribute for attribute in REQUIRED if not hasattr(module, attribute)]
    if missing:
        raise ValueError(f"harness {name!r} lacks {', '.join(missing)}")
    return module


def interpreter(value: str) -> Path:
    """A path as given; a bare name is an environment beside the running
    interpreter's (``baselines`` is ``<prefix>/../baselines/bin/python``)."""

    if "/" in value:
        return Path(value).expanduser().resolve()
    return (Path(sys.prefix).resolve().parent / value / "bin" / "python").resolve()


def resolve(harness, settings, world_size):
    """The case a trainer runs: validated, described, counted."""

    from baselines.common.models import useful_flops_per_token

    geometry = settings.validate(world_size)
    request = settings.request()
    model = request.build(dtype=settings.precision().stepped_dtype)
    case = {
        "schema_version": 1,
        "harness": harness,
        "settings": settings.as_dict(),
        "description": settings.describe(world_size),
        "size": request.entry.size,
        "useful_flops_per_step": useful_flops_per_token(model, geometry.sequence_length)
        * geometry.tokens_per_step,
        "model_summary": request.summary(model),
        "resolved_by": sys.executable,
        "environment": settings.environment(),
    }
    case.update(settings.resolve(world_size))
    return case


def trainer_environment(environment, base=None):
    """``base`` (this process's environment) with the case's variables set,
    and unset where the case says none."""

    result = dict(os.environ if base is None else base)
    for name, value in environment.items():
        if value is None:
            result.pop(name, None)
        else:
            result[name] = value
    return result


def command(harness, python, world_size, settings, output):
    """The command line of one case, as the sweep runs it."""

    return [
        sys.executable,
        "-m",
        "baselines.common.launch",
        "--harness",
        harness,
        "--python",
        str(python),
        "--gpus",
        str(world_size),
        "--output",
        str(output),
        *settings.arguments(),
    ]


def trainer_command(harness, python, world_size, output):
    """The torchrun command line of the harness's trainer."""

    return [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        "--max-restarts=0",
        "-m",
        f"baselines.{harness}.train",
        "--output",
        str(output),
        "--case",
        str(output / "case.json"),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Every other flag is the harness's settings.",
    )
    parser.add_argument("--harness", required=True, help="baselines.<harness>")
    parser.add_argument("--gpus", type=int, required=True, help="torchrun world size")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--python", help="the trainer's interpreter, a path or an environment name"
    )
    known, rest = parser.parse_known_args(argv)
    if known.gpus < 1:
        parser.error("the GPU count must be positive")
    try:
        module = harness_module(known.harness)
    except ValueError as error:
        parser.error(str(error))
    flags = argparse.ArgumentParser(prog=f"{parser.prog} --harness {known.harness}")
    module.add_arguments(flags)
    settings = module.from_namespace(flags.parse_args(rest))
    python = interpreter(known.python or module.ENVIRONMENT)
    if not os.access(python, os.X_OK):
        parser.error(f"{python} is not an interpreter")
    output = known.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    case = resolve(known.harness, settings, known.gpus)
    write_json(output / "case.json", case)
    trainer = trainer_command(known.harness, python, known.gpus, output)
    sys.stdout.flush()
    os.execve(trainer[0], trainer, trainer_environment(case["environment"]))


if __name__ == "__main__":
    main()
