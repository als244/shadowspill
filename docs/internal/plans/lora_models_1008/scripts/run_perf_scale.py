"""Run/resume performance-gate-scale full/LoRA comparisons."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import threading

import psutil

PLAN = Path(__file__).resolve().parents[1]
ROOT = next(p for p in PLAN.parents if (p / "workloads").is_dir())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", type=Path, default=PLAN / "evidence/performance-gate-scale")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--variants", nargs="+", choices=("auto", "save", "recompute"), default=["auto"])
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    progress_file = args.outdir / "progress.json"
    progress = json.loads(progress_file.read_text()) if args.resume and progress_file.exists() else {}
    cases = [dict(family=family, mode=mode, variant=variant)
             for family in ("llama3", "qwen35", "olmoe")
             for variant in args.variants
             for mode in ("full", "lora")]
    for index, case in enumerate(cases, 1):
        name = "-".join(case.values())
        output = args.outdir / name
        output.mkdir(exist_ok=True)
        result_file = output / "result.json"
        if result_file.exists():
            if not args.resume:
                parser.error(f"{name} already exists; use --resume or a new output directory")
            result = json.loads(result_file.read_text())
            expected = {**case, "rank": 32, "factor_dtype": "float32", "execution_gib": 16,
                        "spill_gib": 112, "groups": 3, "steps_per_group": 4, "warmup": 2}
            if any(result["config"].get(key) != value for key, value in expected.items()):
                parser.error(f"{name} has incompatible settings")
            if result["status"] == "passed":
                print("SKIP", name, "already passed", flush=True)
                continue
        started = datetime.now(timezone.utc).isoformat()
        print(f"CASE {index}/{len(cases)} START {started} {name}", flush=True)
        command = [sys.executable, "-u", str(PLAN / "scripts/perf_scale_lora.py"), "--outdir", str(output)]
        for key, value in case.items():
            command.extend(["--"+key, value])
        (output / "command.json").write_text(json.dumps(command, indent=2)+"\n")
        stopped = threading.Event()
        memory = {"peak_tree_rss_bytes": 0, "peak_tree_pss_bytes": 0}
        with (output / "console.log").open("w", buffering=1) as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            def monitor():
                while not stopped.is_set():
                    try:
                        parent = psutil.Process(process.pid)
                        children = [parent, *parent.children(recursive=True)]
                    except psutil.Error:
                        break
                    rss = pss = 0
                    for child in children:
                        try:
                            usage = child.memory_full_info()
                            rss += usage.rss
                            pss += usage.pss
                        except psutil.Error:
                            pass
                    memory["peak_tree_rss_bytes"] = max(memory["peak_tree_rss_bytes"], rss)
                    memory["peak_tree_pss_bytes"] = max(memory["peak_tree_pss_bytes"], pss)
                    stopped.wait(1)
            watcher = threading.Thread(target=monitor, daemon=True)
            watcher.start()
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                code = process.wait()
            finally:
                stopped.set()
                watcher.join()
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
        (output / "host-memory.json").write_text(json.dumps(memory, indent=2)+"\n")
        record = dict(status="passed" if code == 0 else "failed", exit_code=code, started=started,
                      finished=datetime.now(timezone.utc).isoformat())
        if code == 0:
            result = json.loads(result_file.read_text())
            record.update(step_seconds=result["median_step_seconds"], tokens_per_second=result["tokens_per_second"])
        progress[name] = record
        temporary = progress_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(progress, indent=2)+"\n")
        temporary.replace(progress_file)
        print("CASE END", name, json.dumps(record), flush=True)
        if code:
            raise SystemExit(code)
    print("ALL CASES PASSED", len(cases), flush=True)


if __name__ == "__main__":
    main()
