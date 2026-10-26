"""Verify completed winners against their portable Plan, without importing Torch."""

from hashlib import file_digest

from benchmarking.quickstart.phases.common import plan_identity, read
from shadowspill import Plan


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def completed(root, *, budget_gib=None, world_size=None):
    request = read(root / "request.json")
    index = read(root / "programs/index.json")
    world = index.get("world_size", 1)
    if world_size is not None and world_size != world:
        raise ValueError("world size differs from the saved Program")
    selected = []
    for winner in read(root / "plans/index.json"):
        if budget_gib is not None and winner["budget"] != int(budget_gib * (1 << 30)):
            continue
        result_path = (
            root / "execution" / winner["file"].removesuffix(".ssp") / "result.json"
        )
        if not result_path.exists():
            continue
        result = read(result_path)
        if result["plan_digest"] != winner["plan_digest"]:
            raise ValueError("completed result belongs to a different Plan")
        ranks = result["per_rank"]
        if sorted(row["logical_rank"] for row in ranks) != list(range(world)) or any(
            not row["steps"] for row in ranks
        ):
            raise ValueError("completed result lacks steps from every logical rank")
        plan = Plan.load(root / "plans" / winner["file"])
        if plan_identity(plan) != winner["plan_digest"]:
            raise ValueError("saved Plan differs from its planning index")
        parallelism = plan.program.parallelism
        if (1 if parallelism is None else parallelism.world_size) != world:
            raise ValueError("saved Plan has a different worker count")
        selected.append(winner)
    if not selected:
        raise ValueError("requested budgets have no completed winners")
    return request, index, sorted(selected, key=lambda row: row["budget"])


def fingerprint(root, request, index, winner, args):
    return {
        "plan_sha256": digest(root / "plans" / winner["file"]),
        "plan_digest": winner["plan_digest"],
        "candidate": winner["candidate"],
        "factory": request["factory"],
        "factory_args": request.get("factory_args", {}),
        "device": request["device"],
        "rank_map": request.get("rank_map"),
        "world_size": index.get("world_size", 1),
        "warmup_steps": args.warmup_steps,
        "capture_steps": args.capture_steps,
        "capture_settle_seconds": args.capture_settle_seconds,
        "gpu_metrics_frequency": args.gpu_metrics_frequency,
        "gpu_metrics_devices": "cuda-visible",
        "capture_protocol": "saved_plan_cpu_barriers_v1",
    }


def verified_capture(case, inputs):
    path = case / "validation.json"
    if not path.exists():
        return False
    previous = read(path)
    if previous["inputs"] != inputs:
        raise ValueError(f"existing capture has different inputs: {case}")
    for key in ("report", "sqlite"):
        if digest(case / previous[key]) != previous[f"{key}_sha256"]:
            raise ValueError(f"saved capture changed: {case / previous[key]}")
    return True
