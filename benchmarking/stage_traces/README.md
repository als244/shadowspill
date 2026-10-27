# Isolated compiled stage tracing

Capture the same partitioned forward/backward graph pairs used by quickstart,
deduplicate equivalent stages, and trace each unique pair on one GPU. A process
per visible GPU drains the shared queue. Embedding-containing stages are omitted;
optimizer stages are never captured. Both saved-activation and recompute variants
are available. The worker uses the same packaged task compiler as `build_*`.
Backward consumes the actual forward's saved tensors.

From an allocation, run for example:

```sh
python -m benchmarking.stage_traces.sweep \
  --factory workloads.models.glm53_flash.stage_trace:workload \
  --factory-args '{"checkpoint":"/path/to/GLM-5.3-Flash","lora_rank":32,"lora_alpha":32}' \
  --sequence-lengths 8192 --batch-sizes 1 2 4 8 \
  --variants save recompute --gpus all \
  --warmup 3 --iterations 5 --output runs/glm-stage-traces
```

The GLM adapter reads checkpoint metadata to reproduce physical shapes and dtypes,
including frozen FP8 experts. It initializes only each isolated stage's inputs;
it does not read the checkpoint payload or materialize the entire real model.
Integer controls come from capture, and unknown controls are rejected. Different
models provide a factory returning `TraceWorkload`; model-specific initialization
recipes are importable callables attached to physical state names.

The run contains `scripts/`, `logs/`, `notes/`, `graphs/`, and `results/`.
Results are grouped by stage, sequence length, batch size, variant, and attempt.
Each attempt retains its command, console log, result/failure, `.nsys-rep`, and
SQLite export. NVTX ranges identify the stage, geometry, iteration, forward,
and backward. Verification checks all expected ranges and recorded CUDA kernels.
CUDA-event timings under Nsight are diagnostic and are not throughput benchmarks.
OOM outcomes remain explicit; no requested geometry is silently reduced.

Use `--gpus 0 1` to select visible ordinals; use `--discover-only` to prepare
the queue. `--resume` requires identical scripts/options; `--retry-failed`
creates new attempts. Every worker uses one Inductor compile thread to avoid
competing subprocess pools. Independent workers must leave
`MLOPS_QUACK_TUNE_DISTRIBUTED` disabled.

The `workloads.models.glm53_flash.stage_trace:workload` adapter accepts
`{"tiny":true}` for protocol qualification with BF16 experts, and
`{"tiny":true,"tiny_expert_storage":"fp8_block"}` for H100 FP8 qualification.
