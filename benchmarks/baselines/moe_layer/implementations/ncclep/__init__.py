"""NCCL EP dispatcher adapter for the shared NVIDIA expert implementation."""

import os
import sys
from pathlib import Path


def build(**configuration):
    # A stale login-shell NCCL_HOME can override the installed JIT headers.
    # Prefer the environment's matching bundled headers when that path is absent.
    configured = Path(os.environ.get("NCCL_HOME", "/nonexistent"))
    bundled = Path(sys.prefix) / "nccl-home"
    if not (configured / "include/nccl.h").is_file():
        if not (bundled / "include/nccl.h").is_file():
            raise RuntimeError("NCCL EP requires NCCL_HOME/include/nccl.h for its JIT")
        os.environ["NCCL_HOME"] = str(bundled)
    from ..nvidia import build as build_nvidia

    implementation = build_nvidia(dispatcher="ncclep", **configuration)
    implementation.metadata["nccl_home"] = os.environ["NCCL_HOME"]
    return implementation
