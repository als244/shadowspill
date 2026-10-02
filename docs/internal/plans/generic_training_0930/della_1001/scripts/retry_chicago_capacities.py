"""Resume the capacity sweep after the two independent fixes.

Each candidate has a fresh artifact store. Kernel compilation caches may be
reused, but all ShadowSpill task profiles and admission results are regenerated.
No training or W&B run is started by this preparation-only experiment.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from chicago_ep2.report_graph_pairs import report
from chicago_ep2.search import launch, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[32768, 65536, 131072, 262144])
    args = parser.parse_args()
    recipe = Path(__file__).with_name("chicago_ep2")
    config = json.loads((recipe / "config.json").read_text())
    root = Path(config["outdir"]).with_name("chicago-ep2-capacity-fixes-1002")
    root.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for tokens in args.tokens:
        if tokens <= 0 or config["tokens_per_step"] % (config["world_size"] * tokens):
            raise ValueError(f"tokens/rank={tokens} must divide the fixed global step")
        directory = root / f"tokens-{tokens}"
        directory.mkdir(parents=True, exist_ok=True)
        result_path = directory / "result.json"
        previous = json.loads(result_path.read_text()) if result_path.exists() else {}
        if previous.get("status") == "passed":
            print(f"RESUME completed tokens/rank={tokens}", flush=True)
            for rank in range(config["world_size"]):
                report(directory / f"rank-{rank:05d}/plan-diagnostics.json")
            outcomes.append(previous)
            continue
        case = {**config, "microbatch_tokens": tokens, "outdir": str(directory)}
        path = directory / "config.json"
        write(path, case)
        started = time.monotonic()
        record = {
            "tokens_per_rank": tokens,
            "status": "running",
            "started_utc": datetime.now(UTC).isoformat(),
        }
        write(result_path, record)
        print(f"START {record}", flush=True)
        code = launch(path, directory, config["world_size"], planning=True)
        record.update(
            status="passed" if code == 0 else "failed",
            exit_code=code,
            seconds=time.monotonic() - started,
            finished_utc=datetime.now(UTC).isoformat(),
        )
        if code == 0:
            record["ranks"] = [
                json.loads((directory / f"rank-{rank:05d}/planning.json").read_text())
                for rank in range(config["world_size"])
            ]
            for rank in range(config["world_size"]):
                report(directory / f"rank-{rank:05d}/plan-diagnostics.json")
        write(result_path, record)
        outcomes.append(record)
        write(root / "progress.json", outcomes)
        print(f"STOP {json.dumps(record)}", flush=True)
    write(root / "completed.json", {"cases": outcomes, "all_feasible": all(row["status"] == "passed" for row in outcomes)})


if __name__ == "__main__":
    main()
