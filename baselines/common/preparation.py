"""Share explicit preparation artifacts while keeping every case independent."""

from __future__ import annotations

import hashlib
import json
import platform
import shlex

from . import launch, write_json


def groups(cases, *, engine, world_size, manifest, output):
    """Group compatible cases; only the execution budget may differ.

    Source and environment identities are generated from the code being run;
    the caller supplies no cache stamp. Other engines and multi-rank runs keep
    their existing case path until they expose a preparation contract.
    """
    module = launch.engine_module(engine)
    if not getattr(module, "SHARED_PREPARATION", False) or world_size != 1:
        return {}
    grouped = {}
    for name, settings in cases:
        if settings.artifact_store == "none" or settings.verify_reference:
            continue
        common = settings.as_dict()
        common.pop("execution_gib")
        identity = {
            "settings": common,
            "world_size": world_size,
            "source": manifest["source"],
            "environment": manifest["environment"],
            "host": platform.node(),
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        group = grouped.setdefault(
            key,
            {
                "identity": identity,
                "members": [],
                "directory": output / "preparations" / key,
            },
        )
        group["members"].append((name, settings))
    return {name: group for group in grouped.values() for name, _ in group["members"]}


def ensure(group, *, engine, python, world_size, timeout):
    """Run one preparation worker to completion before any member case starts."""
    from baselines.sweep import run_case

    root = group["directory"]
    budgets = sorted(
        {
            (
                int(settings.execution_gib * (1 << 30)),
                int(settings.spill_gib * (1 << 30)),
            )
            for _, settings in group["members"]
        }
    )
    artifact = root / "artifact"
    published = root / "prepared.json"
    if published.exists() and (artifact / "training.json").exists():
        result = json.loads(published.read_text())
        if result.get("status") == "completed" and result.get("budgets") == [
            list(b) for b in budgets
        ]:
            return artifact
    settings = max(
        (settings for _, settings in group["members"]), key=lambda s: s.execution_gib
    )
    # This runtime is only for profiling/preparation. It exits before a case
    # opens the smaller physical pool the case's result will report.
    attempts = list(root.glob("attempt-*"))
    number = max((int(p.name.split("-")[-1]) for p in attempts), default=0) + 1
    worker = root / f"attempt-{number:03d}"
    completed = worker / "preparation-result.json"
    worker.mkdir(parents=True, exist_ok=True)
    request = {
        "directory": str(artifact),
        "budgets": [list(b) for b in budgets],
        "identity": group["identity"],
    }
    write_json(worker / "request.json", request)
    command = launch.command(engine, python, world_size, settings, worker)
    command += ["--preparation-request", str(worker / "request.json")]
    (worker / "command.txt").write_text(shlex.join(command) + "\n")
    print(
        f"PREPARATION {len(group['members'])} cases: {worker / 'console.log'}",
        flush=True,
    )

    def classify(attempt, returncode, world, settings):
        if returncode == 0 and completed.exists():
            return {"status": "completed"}
        return {"status": "failed", "reason": f"preparation exited {returncode}"}

    outcome = run_case(
        command, worker, settings, world_size, timeout, classify_result=classify
    )
    if outcome["status"] != "completed":
        raise RuntimeError(f"shared preparation failed: {worker / 'console.log'}")
    write_json(published, {**json.loads(completed.read_text()), "attempt": worker.name})
    return artifact


def preview(group, *, engine, python, world_size):
    """Write reviewable preparation commands without creating a worker attempt."""
    root = group["directory"]
    artifact = root / "artifact"
    settings = max(
        (settings for _, settings in group["members"]), key=lambda s: s.execution_gib
    )
    budgets = sorted(
        {
            (int(s.execution_gib * (1 << 30)), int(s.spill_gib * (1 << 30)))
            for _, s in group["members"]
        }
    )
    request = root / "preview" / "request.json"
    write_json(
        request,
        {"directory": str(artifact), "budgets": budgets, "identity": group["identity"]},
    )
    command = launch.command(
        engine, python, world_size, settings, root / "preview" / "worker"
    )
    command += ["--preparation-request", str(request)]
    (root / "preview" / "command.txt").write_text(shlex.join(command) + "\n")
