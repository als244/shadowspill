# EP2 full-training Nsight capture at 128K

## Result — October 2

**PASS.** Both ranks restored checkpoint 900, completed three warmup updates,
and captured updates 904–908. The final report is:

`/home/as1669/storage/shadowspill/generic_training_0930/della_1001/chicago-ep2-capacity-fixes-1002/nsys-128k-5steps-1002/trace.nsys-rep`

The 557,803,041-byte report was exported and audited successfully. Its SHA256 is
`f5f7251e78e557feff00ed5cfa00169a194b6af41d708dc26150923f9af31cf9`.

- All ten complete training-step ranges are present: five per rank.
- All 140,350 kernels, 34,720 copies and 15,960 memsets fall entirely inside
  their process's training-step range; there are no uncovered GPU activities.
- Both GPUs have 70,175 kernels and 30 device metrics, with more than ten million
  metric values per GPU at the requested 10 kHz sampling frequency.
- There are 1,990 compiled-call ranges, 60,000 Quack phase ranges and 214,415
  ShadowSpill annotations covering tasks, transfers and allocation handling.

The median captured update was 6.501 seconds for execution, final writeback and
CPU metric collection. The complete outer NVTX range, which also includes input
preparation, had a median duration of 6.728 seconds. These are profiled timings;
the original unprofiled 900-update run had a 6.399-second median execution time.
The capture's per-rank losses are local contributions to the globally normalized
objective; add the two ranks' contributions to obtain the global loss.

The exact command, source hashes, package versions, per-step timing/losses and
audit counts are in `summary.json` and `trace-audit.json` beside the report. A
compact copy is checked in as [evidence/ep2_nsys_128k.json](evidence/ep2_nsys_128k.json).
Only the experiment client and documentation changed; the production training,
planning and layer implementations are the same as the completed run.

## Requested capture

Five training updates of the Chicago-sized BF16 OLMoE with EP2, 128K tokens
per rank per microbatch, four microbatches per rank, and 1,048,576 global token
slots per update. Same 50 GiB execution budget, 6 GiB external headroom and
80 GiB spill per rank as the completed 900-update run.

Restore the step-900 checkpoint into a fresh process, perform three warmup
updates, then capture updates 904–908. The existing LR schedule is preserved.
The source checkpoint and completed W&B runs are unchanged; profiling writes
only to a separate directory. No new checkpoints or W&B runs are created.

ShadowSpill's existing option is `profiler_annotations=True` on the planned
training callable. It labels tasks, compiled calls, transfers and allocations.
Quack/MoonEP phase ranges are already emitted by MLOps. The experiment adds an
outer range per complete update and labels input preparation and metric copies.
`runtime_trace` remains false. Capture starts after setup and warmup, includes
both worker processes and all allocated GPU device metrics at 10 kHz, and stops
after both ranks finish their fifth captured update.

## Launch

Allocation **14882626**, two H100s, 240 GiB host RAM, 32 CPUs, initially requested
for three hours and shortened to **one hour** at the user's request. It ends at
18:19:33 UTC / 14:19:33 Eastern on October 2. Work runs in `codex:0.0`.

```bash
/home/as1669/.conda/envs/shadowspill/bin/python -u \
  docs/internal/plans/generic_training_0930/della_1001/scripts/profile_chicago_ep2.py
```

The launcher accepts `--config`, `--checkpoint`, `--outdir`, `--steps`,
`--warmup` and `--nsys`. It reuses the selected capacity's compilation/profile
store. An existing profile cannot be overwritten.

All outputs live under:

`/home/as1669/storage/shadowspill/generic_training_0930/della_1001/chicago-ep2-capacity-fixes-1002/nsys-128k-5steps-1002/`

The report is `trace.nsys-rep`; `trace.sqlite` is the regenerable audit export.
Console/process logs, exact config and source snapshot, per-rank plans and
profile-step results, and final trace-validation evidence are saved beside it.

To repeat the artifact audit on the head node:

```bash
/home/as1669/.conda/envs/shadowspill/bin/python \
  docs/internal/plans/generic_training_0930/della_1001/scripts/audit_ep2_nsys.py \
  /home/as1669/storage/shadowspill/generic_training_0930/della_1001/chicago-ep2-capacity-fixes-1002/nsys-128k-5steps-1002
```

The head-node watcher polls every ten seconds and wakes the agent on allocation
start, process completion/failure and job expiry. PID and heartbeat are stored
in the output directory. No GPU work is launched automatically by the watcher.

## Capture-start validation

The first attempt stopped after checkpoint restoration and one successful
warmup due to duplicated rank metadata in the experiment logger. Its logs and
source are preserved in `failed-attempt-01/`; the logger was corrected.

The next attempt completed five updates on each GPU, with 70,175 kernels per
GPU and 30 device metrics per GPU. The audit nevertheless rejected it: rank
1's first enclosing NVTX range was absent even though its kernels and task
annotations were present. This report is preserved in
`incomplete-nvtx-attempt-02/`.

That run called `cudaProfilerStart()` only on rank 0. The corrected controller
calls it on every rank, then uses the CPU control-group barrier before any
training range is opened. A bounded two-GPU probe verified all ten initial
step ranges. Its evidence is in `capture-control/`. The final capture passed
the complete audit with this controller; rank 0 stops collection only after
both ranks finish all five steps. This changes profiling control only.

Nsight's stop/export took about three minutes after GPU execution had finished.
That export interval is outside the five training-step ranges. Both workers
and the launcher finished normally at 18:02 UTC, within the one-hour allocation.

Reference: [NVIDIA capture-range documentation](https://docs.nvidia.com/nsight-systems/UserGuide/).
The missing-range diagnosis and correction are supported by the local probe
and full trace audit, rather than an assumption about CUDA synchronization.
