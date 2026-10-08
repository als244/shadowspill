# Model organization and optional expert parallelism

## Decision

Keep architecture construction in `workloads/mlops`, with MLOps providing the
expert implementations. ShadowSpill trainer/planner/runtime remain model-agnostic.
Every MLOps MoE architecture uses the same optional EP constructor arguments.
No separate Quack-flavored architecture directory is needed.

Public family modules: `olmoe.py`, `qwen3_moe.py`, `qwen35_moe.py`.
Qwen attention, initialization, decoder and expert math stay in `_qwen_moe/`.
One private construction helper owns EP resource setup and cleanup across families.
It is not retained on models, so capture does not copy CUDA resource objects.
Local models keep their existing state-dict names, initialization and math.
The pure-PyTorch twins remain independent reference implementations.

## Agenda

- [x] Consolidate EP buffer/layer construction and cleanup.
- [x] Add optional EP to the regular MLOps OLMoE; remove duplicate model directory.
- [x] Expose Qwen3MoE and Qwen35MoE public family modules and aliases.
- [x] Update active callers, catalog, and examples.
- [x] Verify local-model numerics and published-architecture reference parity.
- [x] Verify shared/borrowed buffers and cleanup on construction failure.
- [x] Check imports/meta construction without optional GPU dependencies.
- [x] Record validation scope and remaining LoRA work.

## Progress

2026-10-08: Source audit complete. Qwen3 already exists under the size-specific
Qwen30B name; Qwen3.5 shares its decoder helper package. OLMoE duplicates model
construction under workloads/quack. Both EP paths use the same Quack kernels,
but repeat buffer ownership, parameter relocation and router code. This refactor
will consolidate construction without changing those kernels or routing math.

2026-10-08: Implementation and focused verification completed on Chicago.

## Resulting structure

```text
workloads/mlops/
  llama3.py
  qwen35.py
  olmoe.py             # local or EP, one architecture
  qwen3_moe.py         # Qwen3MoE + Qwen3MoEConfig
  qwen35_moe.py        # Qwen35MoE + Qwen35MoEConfig
  _expert_parallel.py # construction, ownership and routed-output helper
  _qwen_moe/          # shared Qwen decoder, attention, experts and initialization
```

- All three MLOps MoE constructors accept optional `ep_group`, one of
  `token_capacity`/`buffer`, device/parameter placement and expert precision.
- One created or borrowed buffer is shared across all layers; publication banks
  are shared too. Model parameters remain distinct. Temporary factories and
  CUDA resource objects are not retained on parent model objects.
- Cleanup handles failures after some layers were constructed, closes layer
  runtimes before owned buffers, preserves borrowed buffers, and is idempotent.
- CPU relocation preserves parameter `requires_grad`, including frozen state.
- Resource allocation uses the explicitly selected compute device.
- Local state keys, initialization, routing normalization, shared-expert math,
  and auxiliary definitions are retained.
- Removed `workloads/quack/`; existing experiment callers import regular OLMoE.
- Public quickstart aliases are `mlops_qwen3moe` and `mlops_qwen35moe`.
- Updated catalog, source links, active Python callers and CLI choices. Historical
  results/notes retain their recorded names and are not rewritten.
- No installed ShadowSpill core or MLOps kernel changes in this refactor.

## Validation

- Exact before/after CPU snapshots: OLMoE, Qwen3 MoE and Qwen3.5 MoE weights,
  logits and all parameter gradients match with zero tolerance.
- 65 workload tests passed, including Transformers Qwen reference parity and
  local PyTorch/MLOps comparisons. One accelerator test deselected.
- EP resource tests use stand-ins for GPU layers/buffers: ownership, one-buffer
  sharing, precision arguments, rank-unique parameter enumeration, parameter
  relocation, and failure cleanup. They do not validate collective kernels.
- Fresh-process discovery and local meta construction pass with MoonEP, Quack,
  Transformer Engine and `mlops.expert_parallel` imports explicitly blocked.
- 21 documentation tests, published preset count inventory, quickstart help,
  Ruff checks and git diff --check pass.
- The first inventory check caught an outdated import in the one-off inventory
  script; fixed to the private Qwen helper path. A subsequent documentation
  check caught a backend-specific term disallowed by the repository's style
  contract; wording now consistently says compute device.
- Fresh multi-GPU/Hopper validation of the consolidated constructor remains
  outstanding. The EP kernels themselves are unchanged.

Evidence: `logs/model-organization-tests.log`, `logs/model-docs.log`,
`evidence/model-organization-tests.xml`, `evidence/model-docs-tests.xml`,
`evidence/model-catalog.json`, and the small ignored local numerical snapshot.

## Remaining broader work

Whole-model LoRA conversion/default target selection and dedicated non-EP expert
LoRA remain tracked in AGENDA.md. The separately tested LoRA head operation is
unchanged by this refactor. Current source changes are uncommitted.
