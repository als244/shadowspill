"""Head-node reservation watcher: wake this thread; never start GPU work silently."""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

PHASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PHASE.parent / "sim_error_0929/scripts"))
import watch_reservation as wake

wake.THREAD = os.environ.get("CODEX_THREAD_ID", wake.THREAD)


async def main():
    previous = None
    while True:
        try:
            result = subprocess.run(
                ["squeue", "-h", "-u", "as1669", "--name=codex-qwen-ep8-1007",
                 "-o", "%i|%T|%N|%L|%S|%R"], capture_output=True, text=True,
                timeout=10, check=True,
            )
            state = result.stdout.strip()
            record = {"utc": wake.now(), "pid": os.getpid(), "state": state,
                      "thread": wake.THREAD, "poll_seconds": 10}
            (PHASE / "evidence/watcher.json").write_text(json.dumps(record, indent=2) + "\n")
            if state != previous:
                print(wake.now(), state, flush=True)
                previous = state
            if "|RUNNING|" in state:
                await wake.contact(
                    f"Qwen EP8 allocation is RUNNING: {state}. Act now in codex:0.0. "
                    f"Read {PHASE}/README.md and PROGRESS.md. "
                    "First verify GPU node prompt, then launch scripts/launch.sh --stage smoke "
                    "and actively supervise both tiny EP8 checks. Then full quickstart: "
                    "Qwen3-30B-A3B and Qwen3.5-35B-A3B, BF16, EP8, global 2^22 tokens, "
                    "seqlen1024, microbatch8/16/32/64K per rank, budgets20/30/40/50/60/70GiB, "
                    "all depth/breadth orderings and resolution plans. No GPU work was "
                    "started automatically. One-hour allocation: resume durable progress."
                )
                (PHASE / "evidence/allocation_wakeup.json").write_text(json.dumps(record, indent=2) + "\n")
                return
        except Exception as error:
            print(wake.now(), repr(error), flush=True)
        await asyncio.sleep(10)


asyncio.run(main())
