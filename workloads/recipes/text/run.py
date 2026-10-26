"""Configured text runs: independent phases, or an ordinary PyTorch reference."""

import argparse
import json
from pathlib import Path

from shadowspill.training import config
from workloads.recipes.phases import run_phases


def run_config(path, overrides=(), *, phase="all"):
    raw = config.load(path, list(overrides))
    if raw.get("backend"):
        raise ValueError("use engine='pytorch' or engine='shadowspill', not backend")
    if raw.get("engine") == "pytorch":
        if phase != "all":
            raise ValueError("ordinary PyTorch reference execution has no saved phases")
        from .reference.run import run_config as run_reference

        return run_reference(path, overrides)
    if raw.get("engine", "shadowspill") != "shadowspill":
        raise ValueError("use engine='shadowspill' with explicit phase settings")
    if phase not in {"all", "build", "profile", "plan", "execute"}:
        raise ValueError(f"unknown phase: {phase}")
    if phase in {"all", "execute"} and not (raw.get("initialize") or raw.get("resume")):
        raise ValueError("supply initialize=(chunk, destination) or resume=checkpoint")
    if raw.get("parameter_metrics") is not None:
        raise ValueError(
            "use parameter_norms_every; raw metric callbacks are not tasks"
        )
    if raw.get("startup_diagnostics"):
        raise ValueError("profile the saved Plan explicitly; no hidden startup updates")
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
    run_phases("workloads.recipes.text.experiment", root, phase)
    if phase in {"all", "execute"}:
        from .experiment.results import load

        return load(root / "result.json")
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument(
        "--phase", choices=["all", "build", "profile", "plan", "execute"], default="all"
    )
    args = parser.parse_args()
    run_config(args.config, args.overrides, phase=args.phase)


if __name__ == "__main__":
    main()
