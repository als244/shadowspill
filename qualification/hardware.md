# Qualification hardware and precision

The launchers resolve hardware defaults before model construction. Cases print
and save the selected dtypes, budgets and geometry.

| Setting | SM80 and newer | Older GPUs |
| --- | --- | --- |
| Model and gradients | BF16 | FP16 |
| AdamW moments | BF16 | FP32 |
| Master weights | None | None |
| Numerical device allowance | Llama/Qwen: 10 GiB; OLMoE: 8 GiB | 3 GiB |
| Performance device allowance | 16 GiB | 10 GiB |
| Performance spill capacity | 112 GiB | 112 GiB |

Older-GPU numerical presets use four layers and vocabulary size 8,192, retaining
attention/expert widths. Explicit model fields and budgets override defaults.
Full numerical geometries and reference identities are described in
[numerical qualification](numerical/README.md).

Both matrices accept independent precision overrides:

| Option | Values |
| --- | --- |
| `--model-dtype` | `float16`, `bfloat16`, `float32` |
| `--master-dtype` | `none`, `float16`, `bfloat16`, `float32` |
| `--grad-dtype` | `parameter`, `float16`, `bfloat16`, `float32` |
| `--opt-state-dtype` | `float16`, `bfloat16`, `float32` |

`parameter` uses the model-weight dtype for gradient accumulation. Precision
changes are part of numerical reference identity. Custom factories own model and
optimizer precision through `--case-option` and can use master/gradient overrides.

Device allowances include process/runtime overhead. `--external-headroom-mib`
reserves 512 MiB by default inside the allowance; zero removes this reservation.
Executor always enforces the admitted budget. Performance floors apply only when
hardware, geometry, precision and budgets match the recorded configuration.

The suite probes dtype support in a child process before collection, leaving its
own allocator uninitialized. Its generic low-precision tests use BF16 when
supported and FP16 otherwise. Explicit pytest `--test-dtype` overrides
`SHADOWSPILL_TEST_DTYPE`, which overrides automatic selection; the choice is
forwarded to CTest workers. BF16-specific CPU fixtures retain their declared dtype.

The numerical gate uses nearest AdamW moment rounding and deterministic kernels
where available. Quickstart and performance workloads use stochastic BF16 moment
rounding; FP16/FP32 moments use nearest rounding. Ordinary model/training defaults
are independent of qualification's older-GPU presets.
