"""Where a host's artifacts live: the run directory alone on one host, a
target directory per hardware when several hosts share one run.

Every host builds, profiles and plans for its own GPUs, since a Program's
executables and profiles belong to the hardware that made them; the ranks
agree on what they exchange (``shadowspill.parallelism.agreement``), not on
their kernels. The build phase names the host's target after its GPU and
records it in ``target.json`` beside the host index, so the phases that run
without a GPU find it too.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path


def gpu_tag() -> str:
    """This host's GPU as a directory name: its product name and compute capability."""
    import torch

    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") + f"_sm{major}{minor}"


def host_index() -> int:
    """Which of the config's hosts this process runs on, as the orchestrator says."""
    return int(os.environ.get("SHADOWSPILL_HOST_INDEX", "0"))


def record(root: Path, options) -> dict:
    """Name this host's target (needs a GPU) and record it for the later phases."""
    if not options.get("hosts"):
        return {"tag": None, "host": 0}
    entry = {"tag": gpu_tag(), "host": host_index()}
    (root / "target.json").write_text(json.dumps(entry, indent=2) + "\n")
    return entry


def read(root: Path, options) -> dict:
    if not options.get("hosts"):
        return {"tag": None, "host": 0}
    path = root / "target.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing: the build phase names this host's target"
        )
    return json.loads(path.read_text())


def artifacts(root: Path, options) -> Path:
    """The directory this host's programs, profiles, calibration and plans live in."""
    entry = read(root, options)
    return root if entry["tag"] is None else root / "targets" / entry["tag"]


def first_rank(options, host: int) -> int:
    """The global rank of a host's first worker."""
    hosts = options.get("hosts") or [{"ranks": options.get("ranks", 1)}]
    return sum(int(item.get("ranks", 1)) for item in hosts[:host])
