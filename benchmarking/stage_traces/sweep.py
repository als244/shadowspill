"""Capture unique stages, then drain a queue with one Nsight process per GPU."""

from benchmarking._serialization import atomic_json

from .arguments import parse_args
from .discovery import discover, prepare_run
from .process import visible_devices
from .queue import run_queue
from .report import write_index


def main():
    args = parse_args()
    options = prepare_run(args)
    devices = visible_devices(args.gpus)
    if len(set(devices)) != len(devices) or not devices:
        raise ValueError("GPU list must be nonempty and unique")
    cases = discover(args, devices[0])
    atomic_json(args.output / "queue.json", {"cases": cases})
    write_index(args.output, cases, [])
    print(
        f"QUEUE {len(cases)} unique stage/variant/geometry traces "
        f"on {len(devices)} GPUs",
        flush=True,
    )
    if not args.discover_only:
        run_queue(cases, devices, options, args.output, retry_failed=args.retry_failed)


if __name__ == "__main__":
    main()
