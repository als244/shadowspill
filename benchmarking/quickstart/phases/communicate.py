"""Calibrate the union of saved candidates' communication, independently of compute."""

from shadowspill import CommunicationInfo, Program
from shadowspill.communication.requests import collective_requests

from .common import paths
from .distributed import workers


def probes(root, index):
    programs = [
        Program.load(root / "programs" / entry["file"])
        for entry in index["candidates"]
        if "failure" not in entry
    ]
    first = programs[0]
    if any(program.parallelism != first.parallelism for program in programs):
        raise ValueError("quickstart candidates must declare the same parallelism")
    requests = {
        request
        for program in programs
        for request in collective_requests(program, None)
    }
    return first, [request.to_dict() for request in sorted(requests)]


def run(request):
    root, index = paths(request)
    if request.get("communication_info"):
        CommunicationInfo.load(request["communication_info"]).save(
            root / "communication.ssp"
        )
        return
    if not any("failure" not in entry for entry in index["candidates"]):
        print("communicate: no built candidates; skipping calibration", flush=True)
        return
    program, collectives = probes(root, index)
    from shadowspill import measure_communication

    with workers(request, index) as group:
        print(
            f"communicate: {len(collectives)} collective probes, {group.count} workers",
            flush=True,
        )
        communication = measure_communication(
            program=program,
            collectives=collectives,
            rank_map=group.rank_map,
            device=request["device"],
            warmup_seconds=request["warmup_seconds"],
            measure_seconds=request["measure_seconds"],
        )
        if group.rank == 0:
            communication.save(root / "communication.ssp")
