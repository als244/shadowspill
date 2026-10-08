# LoRA agenda

## Design and preparation

- [x] Audit dense projections, packed experts, and direct weight readers.
- [x] Audit existing EP LoRA and generic trainable-state handling.
- [x] Compare official Megatron Bridge, Unsloth, and Tinker interfaces.
- [x] Record dedicated expert modules, frozen defaults, and optional head LoRA.
- [x] Use attention/mixers, dense MLPs and routed experts as the default; head/shared experts opt in.
- [x] Prepare an isolated dense replacement/configuration/selection CPU probe.
- [x] Check independent packed-expert factor math against merged-weight gradients.
- [x] Probe generic fused-head gradient-demand handling in eager/AOTAutograd.
- [x] Add the public architecture catalog and rename Qwen 3.5 MoE identifiers.
- [x] Validate catalog links, names, configuration fields and parameter counts.
- [x] Unify optional EP construction across OLMoE, Qwen3 MoE and Qwen3.5 MoE.
- [x] Remove the duplicate Quack OLMoE model and give each Qwen family a public module.
- [ ] Recheck consolidated EP constructors on multi-GPU Hopper hardware.

## Implementation and validation

- [x] Finish existing Chicago/Tubingen qualification before production changes or GPU tests.
- [x] Add and validate separate MLOps LoRA head loss.
- [x] Connect head-module conversion to logits and the appropriate loss operation.
- [x] Add dedicated non-EP expert LoRA with unchanged model call signatures.
- [x] Omit frozen parameter-gradient work inside fused operations; preserve input gradients.
- [x] Connect workload configuration and target presets outside model forward definitions.
- [x] Validate initialization, loading, tied parameters, and full-training overrides.
- [x] Check eager, torch.compile, and ShadowSpill save/recompute with fresh stores.
- [x] Inspect graphpairs and optimizer state for frozen-weight allocations.
- [x] Update public documentation/examples and record validation limits.
- [ ] Validate DP/EP when supported hardware is available.

- [x] Complete 20 BF16 full-training/LoRA save/recompute throughput comparisons.
- [x] Complete 1.18B Llama full/LoRA save/recompute host-memory comparisons.
- [x] Check BF16 whole-model numerical parity in addition to FP32 and kernel checks.
- [x] Re-run qualification after the generic storage fixes.

## Progress

2026-10-08: Design/source audit recorded in DESIGN.md. Both Chicago and Tubingen
passed the fresh-store qualification suite and are running numerical qualification.
No production LoRA files changed yet. Preparation is isolated under this plan.

2026-10-08: Isolated CPU probes passed: 11 dense/expert checks and 8 fused-head
checks, including fullgraph AOTAutograd. These are design probes, not GPU or
ShadowSpill validation. Head return inference required an explicit optional-Tensor
schema; the probe also registers its temporary module in sys.modules for Dynamo.
Production MLOps remains unchanged.

User requested a central model catalog and renamed mlops_qwen35b to
mlops_qwen35moe. Corresponding public Python names are Qwen35MoE/Qwen35MoEConfig.
The naming/docs changes are applied on Chicago and CPU validation is running in
codex:model-catalog. They do not change model math. The existing qualification
run started at 5f8dc13c; new naming/docs validation is recorded separately.

2026-10-08: Catalog/naming checks passed (27 tests, 2.63 s), quickstart --help
exposes mlops_qwen35moe, and git diff --check passes. Meta inventory confirms
all eight preset parameter counts and agreement between PyTorch/MLOps twins.
The catalog is workloads/MODELS.md, linked from the root/workload/training/docs
entry points. Production LoRA implementation remains pending qualification.

2026-10-08: User approved optional output-head LoRA and a separate MLOps
LoRA head-loss operation. Preparing/testing a 2.4 MiB source-only staging copy
while existing qualification retains the installed MLOps source unchanged.
The head defaults to frozen; complete head training by name remains available.

2026-10-08: Staged separate semantic/explicit LoRA head-loss operation and
independent full-logits PyTorch implementation. First CPU pass: 32 passed,
6 GPU checks deferred. All gradient-demand combinations, AOTAutograd fullgraph,
custom-op registration/fake checks, frozen-state optimizer exclusion, chunked
logits, and saved-seed memory passed. Adding a noncontiguous-input contract check
and documentation before installing. Tubingen passed all three baseline gates;
Chicago passed suite/numerical and is running performance.

2026-10-08: Both baseline machines passed suite, numerical, performance. Afterward
the new head loss passed 6 GPU checks (FP32/FP16/BF16, eager/Inductor) and
7 GPU dispatcher/ordinary-head regression checks. A three-update model with
nonzero LoRA factors matched independent PyTorch through both forced save and
forced recompute ShadowSpill execution. Frozen head bytes were unchanged.
Graphpairs are in evidence/head-shadowspill-{save,recompute}/{graphpairs.json,
graphpairs.md}; source verification hashes are in installed-lora-head-files.json.
Installed the 14 tested MLOps source/test/doc files on Chicago, uncommitted.
No ShadowSpill compiler or planner changes were needed. Whole-model targeting
and expert conversions remain separate outstanding work.

2026-10-08: Installed-source follow-up passed 74 CPU checks (6 hardware skips);
git diff --check is clean. Public MLOps import resolves to the Chicago checkout.
Detailed operation/memory/validation notes are in LORA_HEAD.md. Whole-model
selection remains explicitly labeled in progress in workloads/MODELS.md.

2026-10-08: Model organization refactor completed and checked on Chicago. See
MODEL_ORGANIZATION.md for the boundary decision, tree, changed interfaces and
validation limits. 65 workload checks and 21 documentation checks pass; all
three local models exactly preserve initialization, logits and parameter grads.
Optional EP imports stay out of CPU/meta construction. Source remains uncommitted.

2026-10-08: Full-model implementation connected outside model forward methods.
CPU validation: 26 full-model checks passed across eight implementations and five
architectures; 64 MLOps head/module/FLOP checks passed. GPU ShadowSpill training
passed Llama3, dense Qwen3.5 and OLMoE save/recompute. Qwen3 MoE exposed an
admission failure: one physical allocation backs three independently bound
persistent aliases. Investigating the generic allocation contract before
continuing; do not mark full-model GPU validation complete yet. Grouped GEMM
tiles now account for small low-rank dimensions and hardware shared-memory limits.

2026-10-08: All 16 full-model GPU training cases passed (eight implementations,
save and recompute, three SGD updates against a reference, frozen state exact).
Two generic ShadowSpill storage issues were fixed: activation cotangents sharing
one allocation now receive independent canonical storage, and empty views retain
proper accounting for their nonempty backing allocation. Focused storage checks
passed (35); workload/documentation regression passed (112). MLOps CPU regression
passed (80) and combined GPU head/grouped/registration regression passed (119).
Grouped FP32 GEMM now honors PyTorch's highest-precision setting. GPU head tests
reset compiler state between independent dtype cases, avoiding the aggregate
Dynamo recompile limit; finite-scale validation also remains traceable.
The 20-case BF16 full-training/LoRA benchmark is running in codex:lora-gpu.

2026-10-08: Steady BF16 comparison completed Llama and dense Qwen save/recompute. Full OLMoE exposed an existing auxiliary-routing backward bug: scatter counted in FP32 but created ones in the ambient default dtype. Both ordinary and sequence auxiliary paths now explicitly create FP32 count increments; four independent-gradient CPU regressions passed. The sweep resumes completed cases without rerunning them.

2026-10-08: Saved neutral programs now have an explicit optimizer audit: trainable parameter input bytes, FP32 gradient bytes and FP32 AdamW moment/int64 counter bytes match the exact trainable tensor inventory (18/18 completed cases so far). This inspects saved program metadata and does not allocate checkpoint copies. Full OLMoE save: 90.85 ms, 8.62 GiB process RSS, 2.03 GiB spill peak; LoRA: 23.07 ms, 5.53 GiB RSS, 0.60 GiB spill peak. Repeated-block backward outputs drop 153.10 to 15.63 MiB. Router auxiliary-gradient/documentation follow-up passed 20 tests (3 GPU cases deselected); Ruff checks for changed LoRA/kernel files and git diff --check pass.

2026-10-08: All 20 steady BF16 comparisons and all four 1.18B Llama comparisons passed. BF16 CPU/GPU end-to-end numerical check initially used the FP32 tolerance and rejected a 0.043% loss difference. Added explicit BF16 loss/state tolerances plus a stricter check on relative L2 error of actual trainable parameter updates (5% cap); frozen state remains bitwise exact. This supplements the existing tight FP32 reference checks rather than replacing them. Investigating one 313 ms step in each 1B LoRA run with GC timestamp instrumentation.

2026-10-08: Eight BF16 meta-initialization/text-recipe tests pass, including FP32 factor storage, frozen base parameters and a full loss/backward after materialization. The final comparison contains 20 reduced-dimension cases plus four 1.18B Llama cases; all 24 optimizer audits pass. Source/docs remain uncommitted; prepared commit scripts include the general router auxiliary-count dtype fix.

2026-10-08: All 16 additional BF16 full-model GPU/reference cases pass, giving 32 cases total including FP32. Each performs three SGD updates with nonzero LoRA B factors and head LoRA, then compares all model state and verifies frozen parameters bitwise. `correctness.json` indexes the evidence. Longer 1B timing runs are stable (40 save steps: 99.03–100.78 ms; recompute: 121.77–123.72 ms). A 231.85 ms generation-2 collection occurred during save warmup; there were no generation-2 collections in either measured window.

2026-10-08: Full qualification suite first run: 1237 passed, 1 skipped, one naming-only failure in a new neutral-layout comment. Reworded the comment to avoid a reserved historical pool name; focused naming test passes. The complete suite/numerical rerun is active under the same lora_models_1008 name. First-failure summary is preserved in logs/qualification-first-failure.txt; the main gate log contains the rerun.

2026-10-08: Qualification rerun suite is green: 1238 passed, 1 skipped, 38 deselected; both CTest groups passed (20/20 and 51/51). Numerical qualification against the established full-model references is running.

2026-10-08: Qualification is complete and green: suite 1238 passed, 1 skipped (8.6 minutes); numerical 5/5 against existing references (15.1 minutes). All source diffs pass git diff --check; prepared commit scripts pass bash syntax checks and have not been executed. Final report: RESULTS.md; full comparison/graphpair tables: COMPARISONS.md and SCALE_COMPARISONS.md; 40-step follow-up: TIMING.md. Future multi-GPU EP/DP and full-model FP8 validation remain explicitly outside these single-GPU results.

## Performance-gate-scale follow-up (2026-10-08)

User requested full-vs-LoRA step throughput at the default 7B–9B gate model sizes.
- [x] Use unchanged throughput manifests: Llama 8K, Qwen 16K, OLMoE 32K tokens/microbatch; all 64K tokens/update, sequence 1K.
- [x] Prepare resumable twelve-case run (three architectures × full/LoRA × save/recompute), fresh process per case and per-case stdout/progress.
- [x] Complete six normal-planner cases and audit optimizer objects/graphpairs. The initial twelve homogeneous save/recompute cases were superseded after diagnosing the baseline policy mismatch; one completed forced-save case is retained.
- [x] Summarize throughput, variance, host peaks and actual pool allocations; mirror evidence.

Both modes use BF16 base weights, gradients and moments, no masters, 16 GiB execution and 112 GiB spill capacity. LoRA uses rank/alpha 32 and its default FP32 factor storage, with frozen embedding/head/router/norm. Two LR=0 accumulated warmup steps precede three groups of four measured updates. Fixed matching spill reservations mean process RSS is mostly the pool capacity: actual peak spill allocation is reported separately. Exact default model definitions are used without dimension overrides. No production changes planned or commits authorized by this benchmark follow-up. Evidence: evidence/performance-gate-scale/.

Harness setup correction: querying torch.cuda device name initialized CUDA before runtime installation. Moved metadata query after runtime construction; no production change. Original startup failure preserved beside the first case.

Priority correction after user's baseline question: the prior default Llama gate measured 17.98s with a mixed plan (144 save, 128 recompute across eight microbatches), predicting 18.51s. Our forced-all-save result is 22.225s, so it is not the correct primary comparator. Retain that completed controlled case, pause homogeneous sweeps, and run six auto-policy cases first (three architectures × full/LoRA). No change to gate model geometry/precision/budgets. Capture transfer deltas and plan summaries for causal comparison. The LoRA save case was interrupted during setup before measurement.

2026-10-08 performance-gate-scale comparison complete: all six normal-planner cases passed (72 measured updates plus 12 warmups). Llama full/LoRA: 18.429/13.489s; Qwen: 19.052/15.013s; OLMoE: 4.760/3.877s. All optimizer inventories and physical checks pass; no measured device allocations, pinned registrations, event creates or event growth. User requested no further investigation/optimization of expected LoRA overhead, so this follow-up ends with reporting. No production changes or commits were made for this benchmark follow-up. Final evidence: PERF_SCALE.md, perf-scale-validation.json, per-case graphpairs and full build stores.

2026-10-08: User approved committing and pushing all related code/docs and updating all four machines. Source changes are grouped into three MLOps and three ShadowSpill commits; a separate documentation commit records compact evidence and reproduction scripts. Existing unrelated local files are preserved. See SYNC.md for commit IDs and synchronization status.
