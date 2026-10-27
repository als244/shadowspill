"""Quickstart command-line spill options, saved without opening a backend."""

import argparse
from pathlib import Path


def add_arguments(parser):
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--ssd-spill", type=Path)
    group.add_argument("--remote-spill", metavar="HOST:PORT")
    parser.add_argument("--ssd-staging-mib", type=int, default=256)
    parser.add_argument("--ssd-chunk-mib", type=int, default=2)
    parser.add_argument("--ssd-queue-depth", type=int, default=16)


def from_arguments(args, capacity):
    """No filesystem/backend import here: even planning may run on another host."""
    arguments = {"capacity": capacity}
    if args.ssd_spill is not None:
        arguments.update(
            directory=str(args.ssd_spill.expanduser().resolve()),
            staging_bytes=args.ssd_staging_mib << 20,
            chunk_bytes=args.ssd_chunk_mib << 20,
            queue_depth=args.ssd_queue_depth,
        )
        return {"type": "shadowspill.ssd:SSDPool", "arguments": arguments}
    if args.remote_spill is not None:
        host, colon, port = args.remote_spill.rpartition(":")
        if not host or not colon or not port.isdigit() or not 1 <= int(port) <= 65535:
            raise argparse.ArgumentTypeError("--remote-spill must be HOST:PORT")
        arguments.update(host=host, port=int(port))
        return {"type": "shadowspill.network:RemotePool", "arguments": arguments}
    return None
