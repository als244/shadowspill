"""Produce comparison tables directly from completed full-model case artifacts."""
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "evidence/full-model-steady-benchmark"
GIB = 1 << 30
MIB = 1 << 20


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scale", action="store_true")
    args = parser.parse_args()
    base = ROOT / "evidence/llama-1b" if args.scale else BASE
    prefix = "scale-" if args.scale else ""
    results, pairs = [], []
    for file in sorted(base.glob("*/result.json")):
        result = json.loads(file.read_text())
        c = result["config"]
        memory = json.loads(file.with_name("host-memory.json").read_text())
        row = dict(family=c["family"], mode=c["mode"], variant=c["variant"],
                   parameters=result["parameters"], trainable=result["trainable"],
                   step_ms=1000*result["median_step_seconds"], tokens_per_second=result["tokens_per_second"],
                   process_peak_rss_gib=result["peak_host_rss_execution_bytes"]/GIB,
                   process_tree_peak_pss_gib=memory["peak_tree_pss_bytes"]/GIB,
                   spill_capacity_gib=c["resolved_spill_gib"],
                   spill_end_allocated_gib=result["pools_after_execution"]["spill"]["allocated_bytes"]/GIB,
                   spill_peak_allocated_gib=result["pools_after_execution"]["spill"]["peak_allocated_bytes"]/GIB,
                   execution_peak_allocated_gib=result["pools_after_execution"]["execution"]["peak_allocated_bytes"]/GIB)
        results.append(row)
        stages=json.loads(file.with_name("graphpairs.json").read_text())
        for stage in stages:
            for pair in stage["graph_pairs"]:
                # Every artifact has both variants; report the actually executed
                # one here and leave the complete raw inventory in each case.
                if pair["variant"] != c["variant"]:
                    continue
                for direction in ("forward", "backward"):
                    p=pair[direction]
                    if p is None:
                        continue
                    sizes={label:p[key]/MIB for label,key in
                           [("input_mib","input_allocation_bytes"), ("mutated_mib","mutation_allocation_bytes"),
                            ("output_mib","output_allocation_bytes"), ("workspace_mib","task_workspace_bytes")]}
                    pairs.append(dict(family=c["family"], mode=c["mode"], variant=c["variant"],
                                      stage=stage["unique_stage_id"], occurrences=stage["occurrence_count"],
                                      direction=direction, **sizes, total_mib=sum(sizes.values()),
                                      runtime_ms=p["runtime_ns"]/1e6, unstable=p["timing_unstable"]))
    for name, rows in [("comparison",results),("graphpairs",pairs)]:
        if rows:
            with (ROOT/f"{prefix}{name}.csv").open("w") as f:
                w=csv.DictWriter(f, fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    table=["# Full-model LoRA comparisons", "", "BF16 base/compute; rank 32, alpha 32, FP32 factors/gradients/AdamW moments; no masters.",
           "4-layer reduced-dimension models, width 768, vocabulary 32768, 2048 tokens, sequence length 512.",
           "MoE: 32 experts, top-4, expert hidden width 512. These are architecture-preserving benchmarks, not published 30B/35B model sizes.",
           "At least 5 exact-task warmups and 1 second at lr=0; 10 measured training updates. Fresh per-case artifact stores.", "",
           "| Architecture | Mode | Variant | Trainable M | Step ms | tok/s | Peak process RSS GiB | Peak tree PSS GiB | Spill reserved / peak allocated GiB |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    if args.scale:
        table[3:5] = ["Llama 3 numerical preset: 12 layers, width 2048, FFN 7168, vocabulary 128256; 1.180B base parameters.", "2048 tokens per step, sequence length 512; full training and rank-32 LoRA use the same data and initialization."]
    for p in results:
        table.append(f"| {p['family']} | {p['mode']} | {p['variant']} | {p['trainable']/1e6:.3f} | {p['step_ms']:.2f} | {p['tokens_per_second']:,.0f} | {p['process_peak_rss_gib']:.2f} | {p['process_tree_peak_pss_gib']:.2f} | {p['spill_capacity_gib']:.0f} / {p['spill_peak_allocated_gib']:.2f} |")
    table += ["", "RSS is the case process peak, including its pinned pool; PSS is a 250 ms sampled process-tree peak that apportions shared pages and includes compiler children. These are different scopes.",
              "Spill capacity follows the same tensor-count formula for each mode and is reserved up front. Report it separately from live allocation evidence; RSS reductions include smaller reservations.",
              "", "## Individual graph pairs", "", "All sizes are MiB. Total is input + mutated + output + workspace; totals across separate tasks are not a simultaneous model peak.",
              "Occurrences indicate how often a structurally identical stage appears. Stage IDs are local to the case.", "",
              "| Architecture | Mode | Variant | Stage × occurrences | Pass | Input | Mutated | Output | Workspace | Total | ms |",
              "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for p in pairs:
        table.append(f"| {p['family']} | {p['mode']} | {p['variant']} | {p['stage']} × {p['occurrences']} | {p['direction']} | " +
                     " | ".join(f"{p[k]:.2f}" for k in ["input_mib","mutated_mib","output_mib","workspace_mib","total_mib","runtime_ms"]) + " |")
    (ROOT/("SCALE_COMPARISONS.md" if args.scale else "COMPARISONS.md")).write_text("\n".join(table)+"\n")
    print(f"Reported {len(results)} complete cases and {len(pairs)} graph-pair directions")


if __name__ == "__main__":
    main()
