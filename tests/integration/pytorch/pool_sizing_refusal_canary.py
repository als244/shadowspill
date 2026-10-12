"""Fresh-process canary: a budget that leaves no device pool says why."""

from __future__ import annotations

import sys
from pathlib import Path

from shadowspill.pytorch.frontend import PyTorchFrontend
from shadowspill.runtime.bootstrap import RuntimeInstallError, install_runtime
from tests.integration.pytorch.runtime_helpers import two_pool_topology

MIB = 1 << 20


def main() -> int:
    # A CUDA context alone holds several hundred MiB, so no device leaves a
    # pool inside 600 MiB once 512 MiB of headroom is reserved.
    try:
        install_runtime(
            Path(sys.argv[1]).resolve(),
            frontend=PyTorchFrontend(),
            device_ordinal=0,
            device_budget_bytes=600 * MIB,
            external_headroom_bytes=512 * MIB,
            reject_overbudget=True,
            **two_pool_topology(1 * MIB),
            worker_poll_nanoseconds=10_000,
        )
    except RuntimeInstallError as error:
        if "no room for a device pool" not in str(error):
            raise AssertionError(f"the refusal does not say why: {error}") from error
        return 0
    raise AssertionError("a budget the process already fills was accepted")


if __name__ == "__main__":
    raise SystemExit(main())
