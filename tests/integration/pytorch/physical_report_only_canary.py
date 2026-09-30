"""Explicit reporting mode works with either zero or positive headroom."""

import subprocess
import sys

from tests.integration.pytorch.physical_failure_canary import main

if __name__ == "__main__":
    if len(sys.argv) == 3:
        raise SystemExit(main(report_only=True, headroom_mib=int(sys.argv[2])))
    for headroom in (0, 256):
        subprocess.run(
            [sys.executable, __file__, sys.argv[1], str(headroom)], check=True
        )
