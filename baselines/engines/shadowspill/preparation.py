"""Compose existing phase workers; compatible profiles can be replanned on CPU."""

from hashlib import sha256
from pathlib import Path
from time import perf_counter

from benchmarking.quickstart.orchestrate import run_request
from benchmarking.quickstart.phases.common import read

GIB = 1 << 30


def identity(settings, world):
    ignored = {
        "execution_gib",
        "spill_gib",
        "artifact_store",
        "host_headroom_gib",
        "steps",
        "warmup_steps",
        "nsys_capture",
        "orderings",
        "resolution_options",
    }
    return {
        "world_size": world,
        "settings": {
            key: value
            for key, value in settings.as_dict().items()
            if key not in ignored
        },
    }


def directory(case, settings, case_file, output):
    if "preparation" in case:
        return Path(case["preparation"]["directory"]).resolve()
    if "prepared" in case:
        return Path(case["prepared"]).resolve()
    if settings.artifact_store in {"default", "none"}:
        return output / "phases"
    # An explicitly selected location still gives independent cases distinct files.
    name = sha256(str(case_file).encode()).hexdigest()[:16]
    return Path(settings.artifact_store).expanduser().resolve() / name


def prepare(case, settings, case_file, output):
    started = perf_counter()
    root = directory(case, settings, case_file, output)
    world = case["description"]["geometry"]["world_size"]
    budgets = case.get("preparation", {}).get(
        "budgets", [[int(settings.execution_gib * GIB), int(settings.spill_gib * GIB)]]
    )
    if {spill for _, spill in budgets} != {int(settings.spill_gib * GIB)}:
        raise ValueError("one preparation request uses a common spill budget")
    request = {
        "factory": "baselines.engines.shadowspill.factory:experiment",
        "factory_args": {"case_file": str(case_file)},
        "output_dir": str(root),
        "device": "cuda:0",
        "search_budgets": [execution for execution, _ in budgets],
        "spill_bytes": int(settings.spill_gib * GIB),
        "run_budgets": [],
        "steps": settings.steps,
        "communication_info": None,
        "rank_map": None,
        "warmup_seconds": 0.001 if settings.verify_reference else 0.05,
        "measure_seconds": 0.002 if settings.verify_reference else 0.18,
        "search_workers": settings.cpu_threads,
        "external_headroom_bytes": int(settings.external_headroom_gib * GIB),
        "plots": False,
        "orderings": settings.orderings,
        "resolution_options": settings.resolution_options,
        "baseline_identity": identity(settings, world),
    }
    previous = {}
    if (root / "request.json").exists():
        previous = read(root / "request.json")
        if previous.get("baseline_identity") != request["baseline_identity"]:
            raise ValueError(
                f"preparation belongs to a different baseline case: {root}"
            )
    built = (root / "programs/index.json").exists()
    profiled = built and (root / "profiled/index.json").exists()
    if "prepared" in case and not profiled:
        raise ValueError(f"prepared case has no completed profiles: {root}")
    planned = (
        profiled
        and (root / "plans/index.json").exists()
        and set(request["search_budgets"]) <= set(previous.get("search_budgets", ()))
        and all(
            request[key] == previous.get(key)
            for key in (
                "spill_bytes",
                "orderings",
                "resolution_options",
                "search_workers",
            )
        )
    )
    for phase, ready in (
        ("build", built),
        ("profile", profiled),
        ("communicate", built and (root / "communication.ssp").exists()),
        ("plan", planned),
    ):
        if not ready:
            run_request(request, phase)
    return root, {
        "artifact": str(root),
        "budgets": budgets,
        "preparation_seconds": perf_counter() - started,
        "reused_computation": profiled,
        "reused_planning": planned,
    }
