"""Resume full-model cases without repeating successful work; flush every case."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
import threading
import time

import psutil
from pathlib import Path
import subprocess
import sys

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "workloads").is_dir())
PLAN = ROOT / "docs/internal/plans/lora_models_1008"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("smoke", "benchmark", "scale"), default="smoke")
    parser.add_argument("--outdir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "float16"))
    args = parser.parse_args()
    outdir = args.outdir or PLAN / "evidence" / ("full-model-" + args.suite)
    outdir.mkdir(parents=True, exist_ok=True)
    families = ["llama3"] if args.suite == "scale" else ["llama3", "qwen35", "olmoe", "qwen3moe", "qwen35moe"]
    cases = []
    for implementation in (["mlops", "pytorch"] if args.suite == "smoke" else ["mlops"]):
        for family in families:
            if implementation == "pytorch" and family in {"qwen3moe", "qwen35moe"}:
                continue
            for mode in (["lora_head"] if args.suite == "smoke" else ["full", "lora"]):
                for variant in ("save", "recompute"):
                    cases.append(dict(implementation=implementation, family=family, mode=mode, variant=variant))
    progress_path = outdir / "progress.json"
    completed = json.loads(progress_path.read_text()) if args.resume and progress_path.exists() else {}
    for index, case in enumerate(cases):
        name = "-".join(case.values())
        destination = outdir / name
        destination.mkdir(exist_ok=True)
        result = destination / "result.json"
        if result.exists():
            if not args.resume:
                parser.error(f"{name} already has a result; use --resume or a new output directory")
            saved = json.loads(result.read_text())
            expected = {**case, "preset": "1b" if args.suite == "scale" else args.suite,
                        "dtype": args.dtype or ("float32" if args.suite == "smoke" else "bfloat16"),
                        "rank": 4 if args.suite == "smoke" else 32,
                        "tokens": 8 if args.suite == "smoke" else 2048,
                        "sequence_length": 8 if args.suite == "smoke" else 512,
                        "steps": 3 if args.suite == "smoke" else 10}
            mismatches = {key: (saved["config"].get(key), value)
                          for key, value in expected.items() if saved["config"].get(key) != value}
            if mismatches:
                parser.error(f"cannot resume {name} with different settings: {mismatches}; use a new output directory")
            if saved.get("status") == "passed":
                print("SKIP", name, "already passed", flush=True)
                completed.setdefault(name, {"status": "passed"})
                continue
        now = datetime.now(timezone.utc).isoformat()
        print(f"CASE {index+1}/{len(cases)} START {now} {name}", flush=True)
        command = [sys.executable, "-u", str(PLAN/"scripts/full_model_lora.py"),
                   "--outdir", str(destination), "--preset", "1b" if args.suite == "scale" else args.suite]
        for key, value in case.items():
            command += ["--"+key.replace("_", "-"), value]
        if args.suite == "smoke":
            command += ["--rank", "4", "--dtype", args.dtype or "float32"]
        else:
            command += ["--dtype", args.dtype or "bfloat16", "--tokens", "2048", "--sequence-length", "512",
                        "--steps", "10", "--warmup", "5", "--execution-gib", "16" if args.suite == "scale" else "12"]
        (destination/"command.json").write_text(json.dumps(command,indent=2)+"\n")
        with (destination/"console.log").open("w", buffering=1) as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            samples = {"peak_tree_rss_bytes": 0, "peak_tree_pss_bytes": 0}
            stopped = threading.Event()
            def monitor():
                while not stopped.is_set():
                    try:
                        parent = psutil.Process(process.pid)
                        children = [parent, *parent.children(recursive=True)]
                    except psutil.Error:
                        break
                    total_rss = total_pss = 0
                    for child in children:
                        try:
                            memory = child.memory_full_info()
                            total_rss += memory.rss
                            total_pss += memory.pss
                        except psutil.Error:
                            pass
                    samples["peak_tree_rss_bytes"] = max(samples["peak_tree_rss_bytes"], total_rss)
                    samples["peak_tree_pss_bytes"] = max(samples["peak_tree_pss_bytes"], total_pss)
                    stopped.wait(.25)
            watcher = threading.Thread(target=monitor, daemon=True)
            watcher.start()
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
            finally:
                stopped.set()
                watcher.join()
        (destination/"host-memory.json").write_text(json.dumps({**samples, "sample_seconds": .25,
            "scope": "case process plus compiler children; PSS apportions shared pages"}, indent=2)+"\n")
        record = {"status": "passed" if code == 0 else "failed", "exit_code": code,
                  "started": now, "finished": datetime.now(timezone.utc).isoformat()}
        if code == 0:
            data = json.loads(result.read_text())
            record.update(seconds=data["median_step_seconds"], tokens_per_second=data["tokens_per_second"],
                          peak_host_rss_bytes=data["peak_host_rss_execution_bytes"])
        completed[name] = record
        (outdir/"progress.json").write_text(json.dumps(completed,indent=2)+"\n")
        print("CASE END", name, json.dumps(record), flush=True)
        if code:
            raise SystemExit(code)
    print("ALL CASES PASSED", len(cases), flush=True)


if __name__ == "__main__":
    main()
