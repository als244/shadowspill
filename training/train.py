"""Train one run from a JSON config.

    python -m training.train <config.json> [key=value ...]

The run's directory holds everything it writes: ``config.json``,
``metrics.jsonl``, ``packing.jsonl`` (the documents each step trained on),
``plan.json`` and ``planning.json`` on ShadowSpill, the checkpoint, and W&B's
files. Launching the same run directory again resumes from its checkpoint.
Overrides replace config values by dotted key (see ``training.config``); the
run's directory is the config's ``run_dir``, or ``run_dir=<dir>`` given here.
"""

from __future__ import annotations

import sys

from training.trainer import Trainer


def main(arguments: list[str]) -> None:
    if not arguments:
        raise SystemExit(__doc__)
    path, *overrides = arguments
    Trainer.from_config(path, overrides).train()


if __name__ == "__main__":
    main(sys.argv[1:])
