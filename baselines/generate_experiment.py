"""Create an experiment's folder from a template: its spec, one matrix per
engine, a README and a record of how it was made, every value written out.

    python -m baselines.generate_experiment PATH --template NAME \\
        --model NAME --sequence-length N --tokens-per-step N \\
        --gpu-memory-gib G --host-memory-gib H [--gpus N] \\
        [--implementation NAME] [--lora JSON] [--checkpoint NAME] \\
        [--corpus NAME] [--setting NAME=JSON ...] [--description TEXT] \\
        [--variable NAME=VALUE ...]

PATH is the new folder's path under ``baselines/experiments``; a folder that
exists is never written over. A template is a folder of
``baselines/experiments/templates`` holding a spec and each engine's matrix,
in which a string ``"@name"`` stands for a value the inputs decide:

* ``@model``, ``@implementation``, ``@lora``, ``@gpus``,
  ``@sequence_length`` and ``@tokens_per_step``, as given;
* ``@checkpoint`` and ``@corpus``, the checkpoint under the ``{models}``
  variable and the corpus under ``{corpora}`` the experiment reads, and
  ``@variables``, the variables they need; each is left out when not given;
* ``@microbatches``, the tokens per microbatch from one sequence up to one
  rank's whole update in powers of two, ``@accumulated_microbatches``, those
  with more than one microbatch per update, and ``@single_microbatch``, the
  one with a single microbatch;
* ``@recompute``, 0 to 1 in quarters, and ``@recomputed``, without 0;
* ``@execution_budgets``, a quarter, three eighths, a half, three quarters
  and seven eighths of the GPU's memory in whole GiB, ``@summary_budgets``,
  those and the whole GPU, and ``@spill_gib``, the largest multiple of 16 GiB
  within four fifths of the host's memory.

A grid an empty value leaves without a case is dropped. ``--setting``
replaces a shared setting after the template is filled. The experiment is
checked as its dry run would check it, every sweep's cases expanded and
validated, before the folder is written; a checkpoint or corpus is read only
when ``--variable`` gives the variables' values, and otherwise left to the
dry run. The README states the scenario, each engine's matrix grid by grid
with its cases, how to run the experiment with the common runner and where
its results go; ``generation.json`` records the template, the inputs and the
derived values. The folder is the experiment from then on, edited by hand
as it needs.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import tempfile
from fractions import Fraction
from pathlib import Path
from typing import Any

from baselines import experiment, sweep
from baselines.common.launch import engine_module
from baselines.common.provenance import ROOT, source_identity

EXPERIMENTS = ROOT / "baselines" / "experiments"
TEMPLATES = EXPERIMENTS / "templates"

#: What the variables the templates can declare stand for.
VARIABLES = {
    "models": "the directory holding the Hugging Face checkpoints, one directory"
    " per checkpoint",
    "corpora": "the directory holding the tokenized corpora prepare_corpus writes,"
    " one directory per tokenizer family",
}

#: The fractions of the GPU's memory the execution budgets are.
BUDGET_FRACTIONS = (
    Fraction(1, 4),
    Fraction(3, 8),
    Fraction(1, 2),
    Fraction(3, 4),
    Fraction(7, 8),
)

#: A value the template leaves out.
OMIT = object()


def number(value):
    """A whole number as an integer, any other as a float."""

    return int(value) if float(value).is_integer() else float(value)


def derived(args) -> dict[str, Any]:
    """Every value a template's ``@name`` can stand for."""

    tokens_per_rank, remainder = divmod(args.tokens_per_step, args.gpus)
    if remainder or tokens_per_rank % args.sequence_length:
        raise ValueError(
            "the tokens per update must divide into whole sequences on every GPU"
        )
    sizes, size = [], args.sequence_length
    while size <= tokens_per_rank:
        if tokens_per_rank % size == 0:
            sizes.append(size)
        size *= 2
    if sizes[-1] != tokens_per_rank:
        sizes.append(tokens_per_rank)
    budgets = sorted(
        {
            number(round(fraction * Fraction(args.gpu_memory_gib)))
            for fraction in BUDGET_FRACTIONS
        }
    )
    spill = (int(Fraction(4, 5) * Fraction(args.host_memory_gib)) // 16) * 16
    if spill < 16:
        raise ValueError("the host's memory leaves no 16 GiB spill pool")
    used = {
        name: VARIABLES[name]
        for name, given in (("models", args.checkpoint), ("corpora", args.corpus))
        if given
    }
    return {
        "model": args.model,
        "implementation": args.implementation,
        "lora": args.lora,
        "gpus": args.gpus,
        "sequence_length": args.sequence_length,
        "tokens_per_step": args.tokens_per_step,
        "checkpoint": f"{{models}}/{args.checkpoint}" if args.checkpoint else OMIT,
        "corpus": f"{{corpora}}/{args.corpus}" if args.corpus else OMIT,
        "variables": used or OMIT,
        "microbatches": sizes,
        "accumulated_microbatches": sizes[:-1],
        "single_microbatch": sizes[-1:],
        "recompute": [0, 0.25, 0.5, 0.75, 1],
        "recomputed": [0.25, 0.5, 0.75, 1],
        "execution_budgets": budgets,
        "summary_budgets": sorted({*budgets, number(args.gpu_memory_gib)}),
        "spill_gib": spill,
    }


def fill(value, values: dict[str, Any]):
    """``value`` with every ``"@name"`` replaced, and what is left out
    removed from the mapping that held it."""

    if isinstance(value, str) and value.startswith("@"):
        if value[1:] not in values:
            raise ValueError(f"the template names {value}, which nothing derives")
        return values[value[1:]]
    if isinstance(value, dict):
        filled = {key: fill(item, values) for key, item in value.items()}
        return {key: item for key, item in filled.items() if item is not OMIT}
    if isinstance(value, list):
        return [fill(item, values) for item in value]
    return value


def without_empty_grids(matrix: dict) -> dict:
    """A matrix without the grids an empty value left with no case."""

    grids = matrix.get("grid")
    if isinstance(grids, list):
        matrix["grid"] = [
            grid for grid in grids if all(values for values in grid.values())
        ]
    return matrix


def dumps(value) -> str:
    """JSON as the repository writes it: two-space indents, and lists of
    plain values on one line."""

    text = json.dumps(value, indent=2)
    return (
        re.sub(
            r"\[\s+([^\[\]{}]*?)\s+\]",
            lambda match: (
                "["
                + ", ".join(part.strip() for part in match.group(1).split(","))
                + "]"
            ),
            text,
        )
        + "\n"
    )


def setting_value(text: str):
    """A ``--setting`` value: JSON where it reads as JSON, a string else."""

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def checked(folder: Path, values: dict[str, str] | None):
    """The experiment in ``folder`` loaded and every sweep's cases expanded
    and validated, as its dry run does; ``(spec, {sweep: cases})``. Without
    the variables' values, the settings that need them are left out."""

    spec_path = folder / "experiment.json"
    spec = json.loads(spec_path.read_text())
    declared = spec.get("variables", {})
    if declared and not values:
        spec["settings"] = {
            key: value
            for key, value in spec["settings"].items()
            if not (isinstance(value, str) and experiment.VARIABLE.search(value))
        }
        spec.pop("variables")
        spec_path = folder / ".checked.json"
        spec_path.write_text(json.dumps(spec))
    try:
        loaded = experiment.load_spec(spec_path, folder / ".output", values or None)
    finally:
        (folder / ".checked.json").unlink(missing_ok=True)
    cases = {}
    for item in loaded["sweeps"]:
        matrix = experiment.sweep_matrix(
            experiment.matrix_source(item),
            loaded["settings"],
            experiment.variable_values(loaded),
        )
        settings_type = engine_module(item["engine"]).Settings
        cases[item["name"]] = sweep.expand(matrix, loaded["gpus"], settings_type)
    return loaded, cases


def grid_lines(matrix: dict, shared: set[str]) -> list[str]:
    """Each grid of a matrix as ``axis: values`` with its case count, the
    shared settings left out."""

    grids = matrix.get("grid", {})
    grids = grids if isinstance(grids, list) else [grids]
    lines = []
    for grid in grids:
        count = 1
        for values in grid.values():
            count *= len(values)
        axes = "; ".join(
            f"`{axis}` {', '.join(json.dumps(value) for value in values)}"
            for axis, values in grid.items()
            if axis not in shared
        )
        lines.append(f"{count} cases: {axes}")
    for case in matrix.get("cases", []):
        lines.append(
            "1 case: "
            + "; ".join(f"`{key}` {json.dumps(value)}" for key, value in case.items())
        )
    return lines


def readme(name: str, relative: Path, folder: Path, args, spec, cases, matrices):
    """The experiment's README."""

    settings = spec["settings"]
    up = os.path.relpath(ROOT / "baselines", folder)
    weights = (
        f"`{settings['checkpoint']}`, read as a Hugging Face checkpoint"
        if "checkpoint" in settings
        else "initialized weights"
    )
    data = (
        f"the corpus `{settings['data']}`" if "data" in settings else "synthetic tokens"
    )
    mode = settings["mode"]
    if mode == "lora":
        mode = "LoRA " + ", ".join(
            f"{key} {value}" for key, value in settings.get("lora", {}).items()
        )
    contract = ", ".join(
        f"{label} {settings.get(key) or 'none'}"
        for key, label in (
            ("model_dtype", "weights"),
            ("master_dtype", "masters"),
            ("grad_dtype", "gradients"),
            ("opt_state_dtype", "AdamW moments"),
        )
    )
    roundings = (settings.get("parameter_rounding"), settings.get("opt_state_rounding"))
    rounding = (
        "stochastic rounding" if "stochastic" in roundings else "rounding to nearest"
    )
    variables = spec.get("variables", {})
    assignments = " ".join(f"--variable {key}=<dir>" for key in variables)
    spec_file = f"baselines/experiments/{relative}/experiment.json"
    command = f"python -u -m baselines.experiment --spec {spec_file}" + (
        f" \\\n  {assignments}" if assignments else ""
    )
    guide = os.path.relpath(EXPERIMENTS / "README.md", folder)
    length, tokens = settings["sequence_length"], settings["tokens_per_step"]
    training = {"full": "full training", "lora": "LoRA"}.get(
        settings["mode"], settings["mode"]
    )
    lines = [
        f"# {name}",
        "",
        "| Model | Training | Sequence length | Sequences per step | World size"
        " | Hardware |",
        "|---|---|---|---|---|---|",
        f"| `{settings['model']}` | {training} | {length:,} tokens"
        f" | {tokens // length} ({tokens:,} tokens) | {spec['gpus']}"
        f" | {args.device}, {args.gpu_memory_gib:g} GiB each;"
        f" {args.host_memory_gib:g} GiB of host memory |",
        "",
        "These define the experiment: every case of every engine shares them, and"
        " they title its figures.",
        "",
        args.description or "A throughput and memory experiment.",
        "",
        f"Generated by `baselines/generate_experiment.py` from the `{args.template}`"
        " template; `generation.json` records the inputs and the values derived"
        " from them. The folder is the experiment: edit it as it needs, and run"
        f" it with the common runner as [the guide]({guide}) describes.",
        "",
        "## Scenario",
        "",
        "| | |",
        "|---|---|",
        f"| Model | `{settings['model']}`, its"
        f" `{settings.get('implementation', 'mlops')}` implementation, {mode} |",
        f"| Weights and data | {weights}; {data} |",
        f"| Geometry | {settings['sequence_length']:,}-token sequences,"
        f" {settings['tokens_per_step']:,} tokens per update, {spec['gpus']}"
        f" GPU{'s' if spec['gpus'] > 1 else ''} |",
        f"| Precision | {contract}; {rounding}; the narrow-gradient fusion as each"
        " engine defaults |",
        f"| Run | {settings.get('warmup_steps', 3)} warmup and"
        f" {settings.get('steps', 5)} measured updates per case;"
        f" {settings.get('cpu_threads', 8)} CPU threads; at most"
        f" {spec.get('case_timeout', 3600):g} s a case |",
        f"| Sized for | {number(args.gpu_memory_gib)} GiB of GPU memory and"
        f" {number(args.host_memory_gib)} GiB of host memory: the ShadowSpill"
        " budgets, its spill pool and the summary's budgets follow from them |",
    ]
    if variables:
        lines.append(
            "| Variables | "
            + "; ".join(f"`{{{key}}}`: {meaning}" for key, meaning in variables.items())
            + " |"
        )
    lines.extend(
        [
            "",
            "## Engines",
            "",
            "Each engine's options are the settings its README documents; a matrix"
            " holds the engine's own, and the shared settings above join every"
            " case.",
            "",
        ]
    )
    shared = set(settings)
    for item in spec["sweeps"]:
        matrix = matrices[item["name"]]
        fixed = ", ".join(
            f"`{key}` {json.dumps(value)}"
            for key, value in matrix.get("defaults", {}).items()
            if key not in shared
        )
        lines.append(
            f"### {item['name']}: {len(cases[item['name']])} cases"
            f" ([options]({up}/engines/{item['engine']}/README.md))"
        )
        lines.append("")
        if fixed:
            lines.append(f"Every case: {fixed}.")
            lines.append("")
        listed = grid_lines(matrix, shared)
        for line in listed:
            lines.append(f"* {line}")
        named = sum(int(line.split(" ", 1)[0]) for line in listed)
        if named != len(cases[item["name"]]):
            lines.append("")
            lines.append(
                f"The grids name {named} cases; the"
                f" {named - len(cases[item['name']])} that two grids share run once."
            )
        lines.append("")
    lines.extend(
        [
            "## Running it",
            "",
            "From the repository's root, in a tmux window, first the dry run, which"
            " validates every case and launches nothing:",
            "",
            "```bash",
            command + " --dry-run",
            "```",
            "",
            "then the same command without `--dry-run`. A run writes everything into"
            " `output/` in this folder, which git ignores. `--engine <engine>` runs one"
            " engine and keeps what the others hold; the same command again resumes,"
            " `--retry-failed` attempts the failed cases again, and `--overwrite`"
            " runs the engines it runs from scratch. The variables' values are"
            " recorded in `output/`, so a resume need not repeat them.",
            "",
            "## Results",
            "",
            "A run ends with `output/summary.md` and the report in `output/report/`."
            " To keep the summary with the experiment:",
            "",
            "```bash",
            f"python -m baselines.summary baselines/experiments/{relative}/output \\\n"
            f"  --publish baselines/experiments/{relative}/results",
            "```",
            "",
            "which writes `results/summary.md` and its figures;"
            f" `python -m baselines.trace baselines/experiments/{relative}/output"
            " <engine> <case>` traces one case under Nsight Systems.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "path", type=Path, help="the folder, under baselines/experiments"
    )
    parser.add_argument("--template", required=True, help="a folder of templates/")
    parser.add_argument("--model", required=True, help="a catalog model")
    parser.add_argument("--implementation", default="mlops")
    parser.add_argument(
        "--lora",
        type=json.loads,
        default={"rank": 16, "alpha": 16.0},
        help="configure_lora's options, for a LoRA template",
    )
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--tokens-per-step", type=int, required=True)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument(
        "--device", required=True, help="the GPU's name, as its driver reports it"
    )
    parser.add_argument("--gpu-memory-gib", type=float, required=True)
    parser.add_argument("--host-memory-gib", type=float, required=True)
    parser.add_argument(
        "--checkpoint", help="the checkpoint's directory under {models}"
    )
    parser.add_argument("--corpus", help="the corpus's directory under {corpora}")
    parser.add_argument(
        "--setting",
        action="append",
        default=[],
        metavar="NAME=JSON",
        help="replace a shared setting; repeatable",
    )
    parser.add_argument(
        "--description", help="what the experiment is for, for its README"
    )
    parser.add_argument(
        "--variable",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="a variable's value for the check, so the checkpoint and corpus are read",
    )
    args = parser.parse_args(argv)
    if args.path.is_absolute() or ".." in args.path.parts or not args.path.parts:
        parser.error("PATH is a folder under baselines/experiments")
    folder = EXPERIMENTS / args.path
    if folder.exists():
        parser.error(f"{folder} exists; an experiment is generated once")
    template = TEMPLATES / args.template
    if args.template.startswith(".") or not (template / "experiment.json").is_file():
        parser.error(f"no template {args.template!r} in {TEMPLATES}")
    if (
        min(args.gpus, args.sequence_length, args.tokens_per_step) < 1
        or min(args.gpu_memory_gib, args.host_memory_gib) <= 0
    ):
        parser.error("counts, lengths and memories are positive")
    values = {}
    for item in args.variable:
        key, equals, value = item.partition("=")
        if not equals or not key or not value:
            parser.error(f"--variable takes NAME=VALUE, not {item!r}")
        values[key] = value
    try:
        inputs = derived(args)
        spec = fill(json.loads((template / "experiment.json").read_text()), inputs)
        for item in args.setting:
            key, equals, value = item.partition("=")
            if not equals or not key:
                raise ValueError(f"--setting takes NAME=JSON, not {item!r}")
            spec["settings"][key] = setting_value(value)
        matrices = {
            item["name"]: without_empty_grids(
                fill(json.loads((template / item["config"]).read_text()), inputs)
            )
            for item in spec["sweeps"]
        }
        with tempfile.TemporaryDirectory(dir=EXPERIMENTS) as scratch:
            staging = Path(scratch) / args.path.name
            staging.mkdir()
            (staging / "experiment.json").write_text(dumps(spec))
            for item in spec["sweeps"]:
                (staging / item["config"]).write_text(dumps(matrices[item["name"]]))
            _, cases = checked(staging, values)
            identity = source_identity(
                ["baselines/generate_experiment.py", "baselines/experiments/templates"]
            )
            record = {
                "schema_version": 1,
                "template": args.template,
                "inputs": {
                    key: value
                    for key, value in vars(args).items()
                    if key not in ("path", "variable")
                },
                "derived": {
                    key: value for key, value in inputs.items() if value is not OMIT
                },
                "cases": {name: len(listed) for name, listed in cases.items()},
                "generated_at": datetime.datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
                "generator": {
                    "git_head": identity["git_head"],
                    "git_dirty": identity["git_dirty"],
                },
            }
            (staging / "generation.json").write_text(dumps(record))
            (staging / "README.md").write_text(
                readme(args.path.name, args.path, folder, args, spec, cases, matrices)
            )
            folder.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(staging), str(folder))
    except ValueError as error:
        parser.error(str(error))
    total = sum(len(listed) for listed in cases.values())
    print(f"{folder}: {total} cases", flush=True)
    for name, listed in cases.items():
        print(f"  {name}: {len(listed)} cases")
    if spec.get("variables") and not values:
        print("  the checkpoint and corpus were not read; the dry run reads them")


if __name__ == "__main__":
    main()
