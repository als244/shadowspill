"""Summarize completed default-model full/LoRA measurements, including partial sweeps."""
import csv
import json
from pathlib import Path

PLAN = Path(__file__).resolve().parents[1]
CASES = PLAN / "evidence/performance-gate-scale"
GIB = 1 << 30


def main():
    results = {}
    for path in CASES.glob("*/result.json"):
        result = json.loads(path.read_text())
        if result.get("status") == "passed":
            results[(result["config"]["family"], result["config"]["mode"], result["config"]["variant"])] = (path, result)
    completed_auto = sum(key[2] == "auto" for key in results)
    text = ["# LoRA at the default performance-gate model sizes", "",
            f"Completed normal-policy cases: **{completed_auto}/6**; additional controlled cases: **{len(results)-completed_auto}**. "
            "Each case runs in a fresh process on Chicago's RTX 5090.", "",
            "## Configuration", "",
            "Unmodified MLOps Llama3, dense Qwen3.5 and OLMoE throughput presets. "
            "All updates contain 65,536 tokens at sequence length 1,024. "
            "Microbatches: Llama 8,192 tokens ×8; Qwen 16,384 ×4; OLMoE 32,768 ×2.", "",
            "Both modes use a 16 GiB execution budget, 112 GiB pinned spill capacity, "
            "BF16 base weights/gradients/AdamW moments and no masters. "
            "LoRA uses rank/alpha 32, BF16 compute with FP32 factor storage; "
            "embedding, head, router, normalization and shared-expert base weights are frozen. "
            "Base weights and input batches are initialized identically before LoRA factors are added.", "",
            "Each run uses two exact accumulated-step warmups at LR=0, then three groups of four "
            "measured updates. The table uses the median group throughput; timings include "
            "required terminal transfers using the performance gate's cycle measurements. "
            "The primary comparison lets the ordinary planner choose save/recompute per task. "
            "Any explicitly labeled save/recompute rows are separate controlled experiments.", "",
            "## Step throughput", "",
            "| Model | Variant | Full step s | LoRA step s | Full tok/s | LoRA tok/s | Speedup |",
            "|---|---|---:|---:|---:|---:|---:|"]
    comparison = []
    for family in ("llama3", "qwen35", "olmoe"):
        for variant in ("auto", "save", "recompute"):
            full = results.get((family, "full", variant))
            lora = results.get((family, "lora", variant))
            if full and lora:
                a, b = full[1], lora[1]
                row = dict(model=family, variant=variant, full_step_seconds=a["median_step_seconds"],
                           lora_step_seconds=b["median_step_seconds"], full_tokens_per_second=a["tokens_per_second"],
                           lora_tokens_per_second=b["tokens_per_second"],
                           speedup=a["median_step_seconds"]/b["median_step_seconds"])
                comparison.append(row)
                text.append(f"| {family} | {variant} | {row['full_step_seconds']:.3f} | {row['lora_step_seconds']:.3f} "
                            f"| {row['full_tokens_per_second']:,.0f} | {row['lora_tokens_per_second']:,.0f} | {row['speedup']:.2f}× |")
    text += ["", "## Why the first 22.2-second Llama result differed from the gate", "",
             "The initial controlled run forced all 272 task choices to save and measured 22.225 s/update. "
             "The recent gate result used 144 save and 128 recompute choices and measured 17.980 s "
             "(18.507 s simulated). Forcing save increased planned fetch traffic from 375.69 to 523.69 GiB "
             "and eviction traffic from 181.11 to 330.95 GiB per update. The completed forced-save result "
             "is retained as evidence, but is not used as the normal-policy baseline.", "",
             "For a large linear projection, full training computes forward, input gradient and weight gradient; "
             "a frozen base weight omits the weight-gradient GEMM. LoRA therefore approaches two-thirds of "
             "the projection FLOPs plus the low-rank products, before recomputation. Attention, activation "
             "kernels, optimizer work and transfers also contribute to total step time.", "",
             "## Memory, model sizes and measurement variation", "",
             "Both modes reserve the same 112 GiB pinned pool. Consequently RSS is not a measure "
             "of their differing live tensor requirements here; the actual peak spill allocation is recorded separately. "
             "Host RSS is the process high-water mark through preparation and execution. "
             "Per-case host-memory.json additionally samples process-tree RSS/PSS including compiler children.", "",
             "| Model | Mode | Variant | Total params B | Trainable M | Host RSS GiB | Spill peak GiB | Group step range s | Graphpairs |",
             "|---|---|---|---:|---:|---:|---:|---|---|"]
    memory_rows = []
    graphpair_rows = []
    for (family, mode, variant), (path, result) in sorted(results.items()):
        groups = [x/result["config"]["steps_per_group"] for x in result["measurements"]["group_seconds"]]
        spill_peak = result["pools_after_execution"]["spill"]["peak_allocated_bytes"] / GIB
        rss = result["peak_host_rss_execution_bytes"] / GIB
        link = path.parent.relative_to(PLAN) / "graphpairs.md"
        text.append(f"| {family} | {mode} | {variant} | {result['parameters']/1e9:.3f} | {result['trainable']/1e6:.2f} "
                    f"| {rss:.2f} | {spill_peak:.2f} | {min(groups):.3f}–{max(groups):.3f} | [table]({link}) |")
        memory_rows.append(dict(family=family, mode=mode, variant=variant, parameters=result["parameters"],
                                trainable=result["trainable"], peak_host_rss_gib=rss, peak_spill_allocation_gib=spill_peak,
                                minimum_group_step_seconds=min(groups), maximum_group_step_seconds=max(groups)))
        for stage in json.loads((path.parent / "graphpairs.json").read_text()):
            for pair in stage["graph_pairs"]:
                for direction in ("forward", "backward"):
                    profile = pair[direction]
                    if profile is None:
                        continue
                    sizes = dict(inputs_bytes=profile["input_allocation_bytes"],
                                 mutated_bytes=profile["mutation_allocation_bytes"],
                                 outputs_bytes=profile["output_allocation_bytes"],
                                 workspace_bytes=profile["task_workspace_bytes"])
                    graphpair_rows.append(dict(family=family, mode=mode, execution_policy=variant,
                                               stage=stage["unique_stage_id"], occurrences=stage["occurrence_count"],
                                               variant=pair["variant"], direction=direction, **sizes,
                                               total_bytes=sum(sizes.values()), runtime_ms=profile["runtime_ns"]/1e6))
    text += ["", "All per-task input, mutation, output and workspace sizes and runtimes are also available "
             "in [perf-scale-graphpairs.csv](perf-scale-graphpairs.csv). These are per-task allocation totals, "
             "not simultaneous whole-model peaks.", "", "## Transfer traffic and selected recomputation", "",
             "| Model | Mode | Recompute choices | Fetch GiB/step | Evict GiB/step | New device allocations during measurement |",
             "|---|---|---:|---:|---:|---:|"]
    for (family, mode, variant), (path, result) in sorted(results.items()):
        if variant != "auto":
            continue
        summary = json.loads((path.parent / "plan-summary.json").read_text())
        delta = result["runtime_delta"]
        count = result["config"]["groups"] * result["config"]["steps_per_group"]
        text.append(f"| {family} | {mode} | {summary['recomputing_group_count']}/{summary['task_alternative_group_count']} "
                    f"| {delta['bytes_fetched']/count/GIB:.2f} | {delta['bytes_evicted']/count/GIB:.2f} "
                    f"| {delta['device_allocations']} |")
    text += ["", "## Reproduction", "", "```bash",
             "bash docs/internal/plans/lora_models_1008/scripts/run_perf_scale.sh", "```", "",
             "The runner resumes completed compatible cases and saves progress after each case. "
             "`scripts/perf_scale_lora.py` accepts CLI settings or a JSON configuration file. "
             "Each case stores the full manifest, parameter inventory, optimizer audit, graphpairs, "
             "build artifacts, detailed timings and console log. These are synthetic-token throughput measurements, "
             "not fine-tuning-quality experiments.", "",
             "[Final validation](perf-scale-validation.json) records all six optimizer/physical audits and "
             "the absence of new device allocations or event-pool growth during measurement. "
             "No production code was changed for this follow-up, and no further LoRA optimization "
             "is included, as requested.", "",
             "Primary evidence and full build stores live on Chicago under "
             "`/home/shein/Documents/grad_school/research/shadowspill/docs/internal/plans/lora_models_1008/`. "
             "Reports, logs and small case evidence are mirrored on Della under the same relative plan directory; "
             "full build stores remain on Chicago.", ""]
    (PLAN / "PERF_SCALE.md").write_text("\n".join(text))
    for name, rows in (("perf-scale-comparison.csv", comparison), ("perf-scale-memory.csv", memory_rows),
                       ("perf-scale-graphpairs.csv", graphpair_rows)):
        if rows:
            with (PLAN / name).open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    print("Reported", len(results), "completed cases", flush=True)


if __name__ == "__main__":
    main()
