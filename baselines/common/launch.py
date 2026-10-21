"""Launch one case of any engine.

    python -m baselines.common.launch --engine NAME --gpus N --output DIR \\
        [--python PATH] <the engine's settings flags>

Every engine starts here, from the sweep or by hand. The settings are
validated in this interpreter, the one that runs the sweep, which can
import the catalog with its implementations, readers and LoRA; the case is
then resolved: the row's description, the model's catalog size, the useful
work of a step counted from the catalog model built on ``meta``, a summary
of what was built, whatever the engine's ``Settings.resolve`` adds (a
recipe engine's Hugging Face configuration), and the environment variables
the case's settings decide (``Settings.environment``). The case is written to
``DIR/case.json`` and torchrun replaces this process, running the engine's
trainer in its own interpreter with ``--output DIR --case DIR/case.json``,
in this process's environment with the case's variables set or unset.
That interpreter is ``--python``, or the environment the engine names
(``ENVIRONMENT``) beside the running one.
"""

from __future__ import annotations

import argparse
import os
import sys
from importlib import import_module
from pathlib import Path

from baselines.common.results import write_json

#: What an engine's settings module provides.
REQUIRED = (
    "Settings",
    "SOURCE_PATHS",
    "ENVIRONMENT",
    "add_arguments",
    "from_namespace",
)


def engine_module(name):
    """The settings module of an engine, checked for what is called."""

    try:
        module = import_module(f"baselines.engines.{name}.settings")
    except ModuleNotFoundError as error:
        raise ValueError(f"no engine {name!r}: {error}") from error
    missing = [attribute for attribute in REQUIRED if not hasattr(module, attribute)]
    if missing:
        raise ValueError(f"engine {name!r} lacks {', '.join(missing)}")
    return module


def interpreter(value: str) -> Path:
    """A path as given; a bare name is an environment beside the running
    interpreter's (``baselines`` is ``<prefix>/../baselines/bin/python``)."""

    if "/" in value:
        return Path(value).expanduser().resolve()
    return (Path(sys.prefix).resolve().parent / value / "bin" / "python").resolve()


def resolve(engine, settings, world_size):
    """The case a trainer runs: validated, described, counted."""

    from baselines.common.models import useful_flops_per_token

    geometry = settings.validate(world_size)
    request = settings.request()
    model = request.build(dtype=settings.precision().stepped_dtype)
    case = {
        "schema_version": 1,
        "engine": engine,
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


def command(engine, python, world_size, settings, output):
    """The command line of one case, as the sweep runs it."""

    return [
        sys.executable,
        "-m",
        "baselines.common.launch",
        "--engine",
        engine,
        "--python",
        str(python),
        "--gpus",
        str(world_size),
        "--output",
        str(output),
        *settings.arguments(),
    ]


def trainer_command(engine, python, world_size, output):
    """The torchrun command line of the engine's trainer."""

    return [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={world_size}",
        "--max-restarts=0",
        "-m",
        f"baselines.engines.{engine}.train",
        "--output",
        str(output),
        "--case",
        str(output / "case.json"),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        epilog="Every other flag is the engine's settings.",
    )
    parser.add_argument("--engine", required=True, help="baselines.<engine>")
    parser.add_argument("--gpus", type=int, required=True, help="torchrun world size")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--python", help="the trainer's interpreter, a path or an environment name"
    )
    known, rest = parser.parse_known_args(argv)
    if known.gpus < 1:
        parser.error("the GPU count must be positive")
    try:
        module = engine_module(known.engine)
    except ValueError as error:
        parser.error(str(error))
    flags = argparse.ArgumentParser(prog=f"{parser.prog} --engine {known.engine}")
    module.add_arguments(flags)
    settings = module.from_namespace(flags.parse_args(rest))
    python = interpreter(known.python or module.ENVIRONMENT)
    if not os.access(python, os.X_OK):
        parser.error(f"{python} is not an interpreter")
    output = known.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    case = resolve(known.engine, settings, known.gpus)
    write_json(output / "case.json", case)
    trainer = trainer_command(known.engine, python, known.gpus, output)
    sys.stdout.flush()
    os.execve(trainer[0], trainer, trainer_environment(case["environment"]))


if __name__ == "__main__":
    main()
