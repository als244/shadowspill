"""Assemble the review summary from measured artifacts; no invented results."""
import json
from pathlib import Path

R=Path(__file__).resolve().parents[1]
def read(path):return json.loads((R/path).read_text())
def case(folder,name):return read(f'evidence/{folder}/{name}/result.json')
families=[('llama3','Llama 3'),('qwen35','Qwen 3.5 dense'),('olmoe','OLMoE'),('qwen3moe','Qwen 3 MoE'),('qwen35moe','Qwen 3.5 MoE')]
correct=read('correctness.json')
max_update=max(x['numerical']['trainable_update_relative_l2_error'] for x in correct if x['numerical'])
max_loss=max(x['numerical']['max_loss_relative_error'] for x in correct if x['numerical'])
s=['# Full-model LoRA validation and performance','',
   '## Result','',
   'LoRA trains end to end through ShadowSpill save and recompute on five architectures: Llama 3, dense Qwen 3.5, OLMoE, Qwen 3 MoE and Qwen 3.5 MoE. The PyTorch Llama/Qwen/OLMoE implementations also pass, giving eight implementations. These checks run on one RTX 5090 on Chicago.', '',
   '- **32 GPU/reference correctness cases passed:** eight implementations × FP32/BF16 × save/recompute, three SGD updates each, nonzero B factors and output-head LoRA.',
   '- Frozen parameters remain **bitwise unchanged**. Tight FP32 tests compare losses, all trainable gradients and updated state; BF16 checks also measure the relative error of parameter updates.',
   f'- Worst BF16 loss error: **{100*max_loss:.3f}%**. Worst BF16 update-vector relative L2 error: **{100*max_update:.2f}%** versus the CPU reference. BF16 tolerances are stated in the harness; they differ from the tight FP32 checks.',
   '- **28 performance cases passed**, including full-vs-LoRA across all five architectures, 1.18B Llama, and optional head LoRA.',
   '- **28 saved-program optimizer audits passed:** optimizer parameter inputs, FP32 gradient inputs and AdamW state exactly match the trainable parameter inventory. Frozen weights have no optimizer gradient or moment allocation.', '',
   'Correctness uses synthetic tokens and repeated updates, and establishes numerical implementation parity rather than downstream fine-tuning quality. [Case index](correctness.json), [optimizer audit](optimizer-audit.json).', '',
   '## Architecture comparison','',
   'Four-layer models, width 768, vocabulary 32768, 2048 tokens per step, sequence length 512. MoE variants use E=32, K=4, H=512. These preserve the architecture but are **not** full published 30B/35B model sizes. BF16 base/compute; rank/alpha 32; FP32 factors, gradients and AdamW moments; no masters. Head/router/norm/shared expert frozen by default.', '',
   '| Architecture | Variant | Full step ms | LoRA step ms | Speedup | Full → LoRA peak host RSS GiB |',
   '|---|---|---:|---:|---:|---:|']
for f,label in families:
 for v in ('save','recompute'):
  a=case('full-model-steady-benchmark',f'mlops-{f}-full-{v}');b=case('full-model-steady-benchmark',f'mlops-{f}-lora-{v}')
  s.append(f"| {label} | {v} | {a['median_step_seconds']*1e3:.2f} | {b['median_step_seconds']*1e3:.2f} | {a['median_step_seconds']/b['median_step_seconds']:.2f}× | {a['peak_host_rss_execution_bytes']/2**30:.2f} → {b['peak_host_rss_execution_bytes']/2**30:.2f} |")
s += ['', '![Host memory and throughput](memory_throughput.png)', '',
      'These are medians of ten steps after at least five exact-step warmups and one second at LR=0. The full table also records sampled process-tree PSS, reserved spill capacity and actual spill allocation: [comparison and all graphpairs](COMPARISONS.md), [CSV](comparison.csv).', '',
      '## 1.18B Llama scale check','',
      '12 layers, D=2048, FFN=7168, vocabulary 128256, 2048 tokens/step, 512-token sequences. A 16 GiB physical execution budget is identical across modes. LoRA trains 15.73M parameters, or 19.90M with head LoRA, compared with 1179.70M under full training.', '',
      '| Mode | Variant | Median step ms | Aggregate tok/s | Peak host RSS GiB | Peak spill allocation GiB |',
      '|---|---|---:|---:|---:|---:|']
for mode,label in [('full','Full'),('lora','LoRA; head frozen'),('lora_head','LoRA including head')]:
 for v in ('save','recompute'):
  a=case('llama-1b',f'mlops-llama3-full-{v}') if mode=='full' else case('llama-1b-timing',f'{mode}-{v}')
  tps=a['config']['tokens']*len(a['steps'])/sum(t['seconds'] for t in a['steps'])
  s.append(f"| {label} | {v} | {a['median_step_seconds']*1e3:.2f} | {tps:,.0f} | {a['peak_host_rss_execution_bytes']/2**30:.2f} | {a['pools_after_execution']['spill']['peak_allocated_bytes']/2**30:.2f} |")
s += ['', 'Full training uses ten measured steps; LoRA rows use the longer 40-step follow-up after at least ten warmups and two seconds. GC remains enabled. Save-mode LoRA ranges 99.03–100.78 ms; recompute ranges 121.77–123.72 ms. An observed 232 ms generation-2 collection during warmup supports startup GC as the likely explanation for isolated stalls in the earlier ten-step runs. [Timing evidence](TIMING.md).', '',
      'Host RSS includes the pinned spill reservation. The same state-size formula selected 22 GiB for full training and 6 GiB for LoRA; those reservations are separate from the **11.98 → 2.47 GiB** measured peak spill allocation. RSS also includes initialization, Python/compiler state and ordinary CPU tensors. [Original scale comparison](SCALE_COMPARISONS.md).', '',
      '### A repeated Llama block: individual graphpairs','',
      'Sizes are MiB. Mutated object size is zero in these functional forward/backward graphs; optimizer mutations are audited separately. Totals are per task, not sums of simultaneously live model memory. Full training has twelve occurrences; the LoRA repeated block has eleven because its first block is grouped with the frozen embedding.', '',
      '| Mode | Variant | Pass | Input | Mutated | Output | Workspace | Total | Runtime ms |',
      '|---|---|---|---:|---:|---:|---:|---:|---:|']
for mode in ('full','lora'):
 for v in ('save','recompute'):
  stages=read(f'evidence/llama-1b/mlops-llama3-{mode}-{v}/graphpairs.json')
  stage=max(stages,key=lambda x:x['occurrence_count']);pair=next(p for p in stage['graph_pairs'] if p['variant']==v)
  for d in ('forward','backward'):
   p=pair[d];sz=[p[k]/2**20 for k in ('input_allocation_bytes','mutation_allocation_bytes','output_allocation_bytes','task_workspace_bytes')]
   s.append(f"| {mode} | {v} | {d} | "+' | '.join(f'{x:.2f}' for x in [*sz,sum(sz)])+f" | {p['runtime_ns']/1e6:.3f} |")
s += ['', 'The large saving is in backward gradient outputs and optimizer state. Frozen weights remain inputs; low-rank projections add some forward work and saved activations. In the repeated block, save-mode forward rises 1.239 → 1.409 ms, backward falls 2.747 → 2.058 ms. The much larger whole-step speedup also reflects reduced optimizer work and host/device state traffic.', '',
      '## Efficiency checks','',
      '- Dense projections compute the small A/B products directly; they never form a full-sized weight delta.',
      '- Routed expert products use grouped GPU kernels, with no Python loop launching a GEMM per expert. Frozen base projections compute needed input gradients and omit weight-gradient GEMMs.',
      '- The bounded head loss omits frozen head gradients. Adding head LoRA costs about 1.8 ms in the 1B save measurement while retaining the memory savings.',
      '- A separate explicit `addmm` epilogue probe showed no consistent gain (about ±2%, identical numerical outputs), so the simpler dense expression is retained. [Probe](evidence/dense-epilogue.json).',
      '- No FP32 copy of frozen BF16 weights or master parameter is introduced by LoRA. Optimizer states belong only to selected trainable tensors.', '',
      '## Fixes and checks','',
      'Whole-model tests found and fixed two generic ShadowSpill storage bugs: shared activation cotangents now receive independent boundary storage when required, and zero-element views retain accounting for their nonempty backing allocations. Neither fix depends on an MoE or LoRA operator name.', '',
      'MLOps grouped GEMM tile selection now respects device shared-memory limits and narrow low-rank dimensions, and FP32 kernels respect the requested matmul precision. Existing MoE auxiliary routing counts now remain FP32 when the global default dtype is BF16/FP16.', '',
      'Focused validation includes 119 MLOps GPU/registration checks, 80 MLOps CPU checks, 35 storage checks, 112 workload/documentation checks, and eight BF16 text-recipe initialization/backward checks. Counts are separate runs and are not an additive unique-test count.', '',
      'Qualification is **green**: suite **1238 passed, 1 skipped**, CTest **20/20 and 51/51**, numerical **5/5 against existing references**, without regenerating references. See the [qualification log](logs/qualification.log).', '',
      '## Reproduction and artifacts','',
      'Primary evidence and complete build stores are on Chicago:', '',
      '```text',
      '/home/shein/Documents/grad_school/research/shadowspill/docs/internal/plans/lora_models_1008/',
      '```', '',
      'From the Chicago checkout and its shadowspill environment, run [scripts/run_full_lora_sweep.py](scripts/run_full_lora_sweep.py) with `--suite benchmark --resume --outdir PATH`. `--suite scale` selects the 1.18B comparison. [full_model_lora.py](scripts/full_model_lora.py) accepts CLI options or `--config FILE.json`; each case saves config, parameters, graphpairs, timings, memory and the full build artifact store. Use the supplied shell launchers for the recorded environment and visible tmux execution.', '',
      '[Source hashes and environment](source-manifest.json) · [Agenda](AGENDA.md) · [Other observations](OBSERVATIONS.md). Small reports and evidence are mirrored on Della under the same relative plan directory; full build stores remain on Chicago.', '',
      '## Scope and review status','',
      'This validates single-GPU full-model FP32/BF16 LoRA. Whole-model EP conversion and full-model FP8 LoRA are not included; the existing QuackMoELoRA/TEMoELoRA modules remain separate. Published 30B/35B sizes and long-run fine-tuning quality are not claimed by these reduced-model checks.', '',
      'The user approved committing and pushing the validated implementation. [Commit and synchronization record](SYNC.md). The scripts group changes by purpose: [MLOps](scripts/commit_mlops.sh), [ShadowSpill](scripts/commit_shadowspill.sh). These scripts record the approved source commit groups.']
(R/'RESULTS.md').write_text('\n'.join(s)+'\n')
print(R/'RESULTS.md')
