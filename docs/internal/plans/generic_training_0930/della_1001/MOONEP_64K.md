# 64K MoonEP failure isolation — 2026-10-02

## Result

The same unmodified MoonEP reproducer passes at 32K and 64K with MoonEP's
declared CuTe DSL **4.4.2**, but fails at 64K with **4.7.1**. These are tests on
the same two H100s, Torch installation, MoonEP revision and synthetic inputs.
Neither MLOps nor ShadowSpill is imported. This isolates a MoonEP/compiler
compatibility problem; it does not establish which upstream component is wrong.

**Adopted fix (10:34 UTC):** following the user's direction, MLOps now compiles
only MoonEP's planner at **PTXAS optimization level 2** with DSL 4.7.1. Quack's
GEMMs and MoonEP's other kernels keep their normal settings. The version/source
checked patch lives in `mlops/expert_parallel/_compat/moonep_compiler.py`, applies
automatically during backend/buffer initialization, and leaves installed MoonEP
files unchanged. No environment downgrade is needed.

Validation: 31 optional EP2 GPU cases passed, including exact 32K/64K transport
with random/skewed routing and buffer reuse; Quack/TE full and LoRA, BF16 and
supported FP8 variants, save/recompute; and shared-bank tests. ShadowSpill's
independent full-model numerical checks also pass for save and recompute:
relative loss difference 7.27e-6, no failed parameter checks. Evidence:
`moonep-planner-o2-validation/` under the storage root below. The complete 16-layer model also passed 64K planning and physical admission
on both ranks at 11:40 UTC, with predicted global-step times 6.564/6.573 s
at the unchanged budgets. Its post-fix training measurement is still pending.
See [full-model results](evidence/capacity_retry_32k64k.json).

| CuTe DSL | Tokens/rank | Outcome on both ranks | Evidence directory |
| --- | ---: | --- | --- |
| 4.7.1 | 32,768 | Dispatch completed | `stock-moonep-14864915-t32768` |
| 4.7.1 | 65,536 | CUDA illegal address | `stock-moonep-14864915-t65536` |
| 4.4.2 | 32,768 | Dispatch completed | `stock-moonep-14864915-dsl442-t32768` |
| 4.4.2 | 65,536 | Dispatch completed | `stock-moonep-14864915-dsl442-t65536` |

All paths above are under
`/home/as1669/storage/shadowspill/generic_training_0930/della_1001/`.
Each contains `console.log`, per-process logs, and `rank-00000.json` /
`rank-00001.json` with incremental phases and versions. The 4.4.2 64K test
completed at **2026-10-02 09:44:34 UTC**. These initial tests check successful
dispatch/synchronization. Follow-up tests at **10:02 UTC** also pass exact
dispatch/combine identity roundtrips and the saved-plan dispatch/combine used
in backward, for **three iterations at both 32K and 64K on both ranks**.
Their logs are `stock-moonep-14864915-dsl442-roundtrip-t{32768,65536}`.
These validate communication values, not expert GEMM or full-layer gradients.

Small copies of the four initial console logs and per-rank JSON records, plus
the exact original reproducer, are archived in
[evidence/moonep_64k_initial/](evidence/moonep_64k_initial/).

## Confirmed isolation

The stock MoonEP public `Buffer.dispatch()` API passes at 32,768 tokens/rank
and fails at 65,536 tokens/rank on two H100s, with E=192, K=4, D=1024,
padding=128, 32 communication SMs and 96 replica slots/rank. Routing is generated
by ordinary PyTorch `randn(...).topk(4)` and `bincount`, independent of our models.

The reproducer imports neither MLOps nor ShadowSpill and asserts that the
MLOps singleton patch is absent. Seven relevant installed MoonEP Python files
match GitHub commit `2bd860b4dd083df62b79d5e916fca71ec5742228` byte for byte.
Source checks are recorded in `evidence/moonep_stock_source.json`.

- Reproducer: `scripts/repro_moonep_64k.py`
- Visible GPU launcher: `scripts/launch_stock_moonep.sh`, in `codex:0.0`
- Initial stock logs: `stock-moonep-14864915-t{32768,65536}` under
  `/home/as1669/storage/shadowspill/generic_training_0930/della_1001/`
- Device allocation: job 14864915 on della-j15g1.

This rules out the MLOps layer implementation as a necessary cause. MoonEP's
installed metadata declares `nvidia-cutlass-dsl==4.4.2`. Our optional backend
installer deliberately overrode that requirement to satisfy current Quack.
The original override lacked this geometry's regression coverage. The new
planner-only compatibility patch and large-token regression address that gap.

## Reproduction

Runtime: job **14864915**, Della node **della-j15g1**, two H100 GPUs,
PyTorch **2.13.0+cu132**, Torch CUDA **13.2**, MoonEP commit
`2bd860b4dd083df62b79d5e916fca71ec5742228` (package version `0.0.1`).
The planner source SHA256 is
`6a9d03a968042d24c514fe4230d6c16f5dc81518b0986637ade6cfd9524025be`.
See [source verification](evidence/moonep_stock_source.json).

Install the older DSL on the **head node**, without changing the main environment:

```bash
PY=/home/as1669/.conda/envs/shadowspill/bin/python
DSL442=/home/as1669/storage/shadowspill/generic_training_0930/della_1001/moonep-declared-dsl442
"$PY" -m pip install --target "$DSL442" --no-deps \
  nvidia-cutlass-dsl==4.4.2 \
  nvidia-cutlass-dsl-libs-base==4.4.2 \
  nvidia-cutlass-dsl-libs-cu13==4.4.2
```

Run from the allocated `codex:0.0` GPU pane:

```bash
# Current environment: 4.7.1, 32K passes / 64K fails.
PROBE_LABEL=dsl471 \
  bash docs/internal/plans/generic_training_0930/della_1001/scripts/launch_stock_moonep.sh

# Same MoonEP source and test, 4.4.2: both sizes pass.
PROBE_DSL_ROOT=/home/as1669/storage/shadowspill/generic_training_0930/della_1001/moonep-declared-dsl442 \
PROBE_LABEL=dsl442 \
  bash docs/internal/plans/generic_training_0930/della_1001/scripts/launch_stock_moonep.sh
```

The portable core command is:

```bash
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  docs/internal/plans/generic_training_0930/della_1001/scripts/repro_moonep_64k.py \
  --tokens 65536 --outdir /path/to/output
```

For 4.4.2 installed with `--target`, prepend
`$DSL442/nvidia_cutlass_dsl/python_packages:$DSL442` to `PYTHONPATH`.
The script asserts that no MLOps/ShadowSpill modules or singleton patch are loaded.
Run without its optional compiler diagnostics for the stock comparison. Add
`--iterations 3 --check-roundtrip` to check communication values and buffer reuse.
No sanitizer was enabled for the four results in the table.

## Comparison with the earlier moe_lab stack

The initial TE environment lock used DSL 4.4.2. The Quack launcher later prepended
its separate dependency directory, using **DSL 4.7.1 / Quack 0.6.5**, pinned to
Quack commit `35266c3298f0e9bf6d5f46c30aace2eaeae517e3`. Torch was
2.13.0+cu132 and Triton 3.7.1, as now. MoonEP was the same revision.

The 2026-09-28 EP8 runs really did pass at 65,536 tokens/rank, including **one
chunk / one buffer**, E=256, K=6, D=4096, expert width=2048, one shared expert.
Their BF16 full-training timings were 77.98 / 143.43 ms forward/backward;
FP8 full-training timings were 61.09 / 97.38 ms. Four-chunk tests also passed,
but those dispatch only 16,384 tokens per chunk and are a weaker 64K control.

Evidence in `dev/moe_lab/` (read only, no production import dependency):

- `moonep_codex_handoff/environment-lock.txt`: original TE lock.
- `scripts/environment.sh`: Quack-specific dependency path override.
- `implementations/QuackMoE/pyproject.toml`: pinned Quack and DSL versions.
- `shadowspill_testing/artifacts/20260922-lora-graphpair-tables/package-versions.json`.
- `benchmarking/results/20260928-ep8-quack-64k-n1m1/{case-plan,results}.json`.

Therefore this is **not a universal 64K failure or simply an upgrade since all
moe_lab tests**. Rank count, expert count, top-k, dimensions and routing differ.
The current reproducer fails before expert GEMMs run, inside MoonEP planning.
Exactly which configuration difference exposes the failure remains under test.

An additional test at **10:10 UTC** runs the stock reproduction using the old
moe_lab Quack environment itself (`source dev/moe_lab/scripts/environment.sh
quack`). It also fails on the current **EP2/E192/K4/D1024/64K** geometry.
Evidence: `stock-moonep-14864915-old-moe-lab-t65536`, followed by
`stock-moonep-14864915-old-moe-lab-exception-t65536` (original CUDA error 700
printed before destructor teardown); launcher:
`scripts/probe_moonep_old_environment.sh`. Thus this failure was latent in that
environment, rather than newly introduced by moving the implementation to MLOps.

With the current DSL 4.7.1 and EP2/K4/D1024/64K held constant, E64 and E128
complete three exact communication roundtrips, while E192 and E256 fail in
planning. E64/K8/D7168/64K also completes three exact roundtrips when using
payload values in multiples of 1/16. E256/K6/D4096/64K also passes that control
on EP2 at 10:15 UTC, in `stock-moonep-14864915-shape-e256-k6-d4096-exact-t65536`.
The first random-valued K8/K6 roundtrip
control used an over-strict bitwise assertion: MoonEP combine prologue rounds
local partial sums back to BF16 before its cross-rank sum. The exact-value
control avoids that rounding ambiguity. These controls support shape-dependent
exposure; they do not identify one source-level cause.

## Using 4.4.2 in the complete backend

MoonEP alone works with the isolated 4.4.2 install. The current Quack 0.6.5
requires DSL >=4.7 and fails immediately with 4.4.2 because
`cutlass.base_dsl.typing.Vector` is absent. Evidence:
`quack-capacity-14864915-quack-dsl442/console.log` under the storage root above.
Quack 0.4.1 declares >=4.4.2, but lacks the current `quack.epilogue.frontend`
API used by our fused kernels. A full-environment downgrade has **not** been
validated or applied. Compatibility work must preserve those computations and
numerical checks; changing the package pin alone is insufficient.

A bounded diagnostic backport replaced Quack's three `Vector` helpers with the
older 0.4.1 implementations. Import then reached another missing API,
`cutlass.base_dsl.enums`. This confirms that fixing one missing import alone
does not establish compatibility. The diagnostic lives in
`scripts/probe_quack_dsl442.py`; log:
`quack-dsl442-vector-backport-import.log`. It has not been installed or enabled
in MLOps. The user selected the smaller planner-only compiler fix for 4.7.1.
Backporting or downgrading Quack is no longer the chosen path.

## Diagnostic evidence and limits

1. Synchronizing each stage in a single packaged layer puts the first failure
   inside `launch_planning`, before dispatch launches. Original router IDs and
   histograms are valid, with exactly 262,144 assignments on each rank.
2. Replaying those inputs directly through the planner also fails. Uniform
   synthetic inputs fail too **without instrumentation**; an early apparent
   routing dependence was confounded by Compute Sanitizer changing execution.
3. Memcheck-instrumented planner runs complete without reported invalid kernel
   accesses. CUDA capability probes generate handled API errors, separate from
   the target failure. Those are disabled in later sanitizer reporting.
4. Racecheck reproduces the illegal address and reports a shared-memory warning
   between loading `owner_remaining` and writing `s_alloc`. Adding a warp sync
   removes neither the underlying failure nor the need for further diagnosis.
5. A kernel stopped after sorting produces exact permutations and expert
   grouping. Allocation totals are 262,144/rank, balancing a 2,142-assignment
   excess. A CPU reconstruction yields valid destination ranks and offsets.
   That observation alone does not prove the full kernel has identical timing.
6. Bounds-check instrumentation on sorted indices or destination ranks makes the
   kernel complete; no check fires. This is diagnostic evidence, not a fix.
7. CUDA core dumps place a hardware MMU fault at the planner's second cross-rank
   barrier (`_common.py:333`), before destination construction. The reported
   address is within the expected metadata range. Core dumps omit the external
   VMM mapping, so inability to read that mapping in CUDA-GDB is inconclusive.
8. Removing the singleton patch, reducing histogram-loop unrolling, and replacing
   multicast plan publication with unicast writes each still reproduce failure.
9. With DSL 4.7.1, limiting **only the planner's** PTX assembler optimization to
   level 0 or 2 makes the stock 64K dispatch complete; default level 3 fails.
   Logs: `stock-moonep-14864915-ptx-O0-t65536` and
   `stock-moonep-14864915-ptx-O2-t65536`. This narrows the compatibility issue;
   the later MLOps fix adopts level 2, with the user's approval. Adding assembly
   memory clobbers alone still failed.
   A complete packaged QuackMoE layer using this diagnostic planner-only setting
   also completes two 64K forward/backward iterations with finite values:
   `quack-capacity-14864915-planner-o2-full-layer`. This is a smoke check, not a
   substitute for independent numerical-reference validation.

The exploratory scripts `probe_quack_capacity.py` and `probe_planning_kernel.py`
modify only their own process for diagnosis. The adopted compatibility patch
is separate from those scripts; installed MoonEP files remain unchanged.

## Next checks

- [x] Independent stock public-API reproduction, 32K control / 64K failure.
- [x] Verify installed source against the pinned upstream revision.
- [x] Repeat using MoonEP's declared CuTe DSL 4.4.2.
- [x] Compare versions and successful 64K configurations with moe_lab artifacts.
- [x] Add repeated numerical dispatch/combine checks with 4.4.2.
- [x] Retain current Quack/DSL versions with the user-selected planner-only fix.
- [x] Identify and validate the smallest justified fix (31 EP2 GPU checks).
- [x] Check packaged-layer forward/backward at 64K with planner O2; independent
      numerical comparisons also pass at the smaller qualification shape.
- [x] Implement generic on-demand saved-value replay; fresh/cached profiling and
      full-model EP2 save/recompute checks pass.
- [x] Finish ShadowSpill suite (1,104 Python + 48 CUDA checks) and numerical gate (5/5 existing H100 references); publish both source fixes.
- [ ] Complete 32K/64K model planning retries, now active.
