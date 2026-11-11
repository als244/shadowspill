"""Frozen candidate descriptions shared by every saved phase."""

import json


def build_candidates(options):
    return tuple(options["candidates"])


def failures(root):
    path = root / "candidate-failures.json"
    return json.loads(path.read_text()) if path.exists() else {}


def available(root, options):
    rejected = failures(root)
    return tuple(c for c in build_candidates(options) if c["path"] not in rejected)


def attempt_candidate(root, candidate, phase, operation, group):
    """Skip only local device exhaustion; a collective failure ends the group."""
    if group.count > 1:
        return operation()
    from orchestration.candidates import attempt

    entry = {"name": candidate["path"]}
    result = attempt(entry, phase, operation, device=0)
    if "failure" in entry:
        rejected = failures(root)
        rejected[candidate["path"]] = entry["failure"]
        (root / "candidate-failures.json").write_text(
            json.dumps(rejected, indent=2) + "\n"
        )
    return result


def program_paths(options, *, root=None):
    kinds = ("training", "evaluation") if options["eval_every"] else ("training",)
    return tuple(
        f"{candidate['path']}/{kind}.ssp"
        for candidate in (
            build_candidates(options) if root is None else available(root, options)
        )
        for kind in kinds
    )
