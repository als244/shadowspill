"""Configured text runs: independent phases, or an ordinary PyTorch reference."""

import argparse
import json
from pathlib import Path

from reference.training.configuration import load_config
from shadowspill.training.preparation.phases import run_phases


def run_config(path, overrides=(), *, phase="all"):
    raw = load_config(path, overrides)
    if raw.get("backend"):
        raise ValueError("use engine='pytorch' or engine='shadowspill', not backend")
    nodes = raw.get("nodes")
    ranks = (
        sum(node.get("ranks", 1) for node in nodes) if nodes else raw.get("ranks", 1)
    )
    if type(ranks) is not int or ranks < 1:
        raise ValueError("ranks must be a positive integer")
    if raw.get("heterogeneous") is not None and (nodes or "ranks" in raw):
        raise ValueError("the heterogeneous section carries its own nodes and shares")
    if raw.get("eval_every") and raw.get("heterogeneous") is None:
        from shadowspill_workloads.inputs.text.units import held_out_units

        # Refused here rather than after build, profile and planning.
        held_out_units(
            raw.get("eval_tokens", raw["max_tokens_per_step"]),
            raw["max_seq_len"],
            [1] * ranks,
        )
    if raw.get("engine") == "pytorch":
        if phase != "all":
            raise ValueError("ordinary PyTorch reference execution has no saved phases")
        if ranks > 1:
            raise ValueError(
                "the PyTorch reference runs one worker; "
                "ranks needs the shadowspill engine"
            )
        from reference.training.run import run_config as run_reference

        return run_reference(path, overrides, options=raw)
    if raw.get("engine", "shadowspill") != "shadowspill":
        raise ValueError("use engine='shadowspill' with explicit phase settings")
    if phase not in {"all", "build", "profile", "communicate", "plan", "execute"}:
        raise ValueError(f"unknown phase: {phase}")
    if phase == "communicate" and ranks == 1:
        raise ValueError("one worker measures its host copies in the profile phase")
    if phase in {"all", "execute"} and not (raw.get("initialize") or raw.get("resume")):
        raise ValueError("supply initialize=(chunk, destination) or resume=checkpoint")
    if raw.get("parameter_metrics") is not None:
        raise ValueError(
            "use parameter_norms_every; raw metric callbacks are not tasks"
        )
    if raw.get("checkpoint_weights", "master") != "master" or not raw.get(
        "checkpoint_include_optimizer", True
    ):
        raise ValueError("saved-Plan resume checkpoints retain full training state")
    root = Path(raw["run_dir"]).resolve()
    if phase in {"all", "build"} and (root / "candidates.json").exists():
        raise ValueError("build exists; choose a later phase or a fresh run directory")
    root.mkdir(parents=True, exist_ok=True)
    raw["run_dir"] = str(root)
    (root / "config.json").write_text(json.dumps(raw, indent=2) + "\n")
    if raw.get("heterogeneous") is not None:
        from shadowspill_workloads.recipes.text.heterogeneous.run import (
            run_heterogeneous,
        )

        run_heterogeneous(root, raw, phase)
    elif nodes:
        from shadowspill_workloads.recipes.text.experiment.nodes import run_nodes

        run_nodes(root, raw, phase, nodes)
    else:
        run_phases(
            "shadowspill_workloads.recipes.text.experiment", root, phase, ranks=ranks
        )
    if phase in {"all", "execute"}:
        from shadowspill_workloads.recipes.text.experiment.results import load

        records = [
            root / "aggregate/wandb_run.json",
            *sorted(root.glob("nodes/*/aggregate/wandb_run.json")),
            root / "wandb_run.json",
        ]
        for record in records:
            if record.exists() and (url := json.loads(record.read_text()).get("url")):
                print(f"W&B run: {url}", flush=True)
                break
        return load(root / "result.json")
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument(
        "--phase",
        choices=["all", "build", "profile", "communicate", "plan", "execute"],
        default="all",
    )
    args = parser.parse_args(argv)
    run_config(args.config, args.overrides, phase=args.phase)


if __name__ == "__main__":
    main()
