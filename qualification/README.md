# Qualification

This tree contains ShadowSpill's reproducible correctness, performance, and
diagnostic gates. Generated checkpoints, traces, planning caches, and reports
are written beneath `qualification/results/`, which is intentionally ignored.

## Numerical matrix

Run the five approximately-1B provider cells in fresh reference and planned
processes:

```bash
python -m qualification.numerical.matrix \
  --build-dir /tmp/shadowspill-build-full \
  --cache-dir qualification/results/cache
```

The matrix supports model subsets, budget overrides, reusable reference
artifacts, and external case factories. See
[`numerical/README.md`](numerical/README.md) for the protocol and individual
worker interface.

```bash
python -m qualification.numerical.matrix \
  --build-dir /tmp/shadowspill-build-full \
  --models qwen35 olmoe \
  --implementations pytorch mlops \
  --budget qwen35=10GiB \
  --budget olmoe=10GiB \
  --reuse-reference \
  --keep-going
```

Models outside the built-in registry use an importable factory and explicit
configuration:

```bash
python -m qualification.numerical.matrix \
  --build-dir /tmp/shadowspill-build-full \
  --models diffusion_transformer \
  --implementations pytorch \
  --budget diffusion_transformer=24GiB \
  --case-factory my_project.shadowspill_cases:build_case \
  --model-config @configs/dit.json \
  --data-geometry @configs/dit_microbatches.json
```

The factory receives the model identity, implementation, seed, model config,
data geometry, and case options. It returns the model, microbatches, objective,
optimizer factory, and provider context used independently by the reference and
planned workers.

## Full-model performance

Run all retained full-model ShadowSpill cells:

```bash
python -m qualification.performance.matrix \
  --output-directory qualification/results/full_model \
  --force-fresh \
  --keep-going
```

Each worker can also be launched directly with
`python -m qualification.performance.run`; see
[`performance/README.md`](performance/README.md).

## Framework-free planning replay

Benchmark canonical PressureFit fixtures without capture, compilation,
profiling, or runtime execution:

```bash
python -m qualification.planning.benchmark \
  qualification/results/numerical_matrix/pytorch_llama3_pressurefit \
  --repeats 3 \
  --output qualification/results/pressurefit_benchmark.json
```

## Diagnostics

- `python -m qualification.diagnostics.step` summarizes serialized
  `StepDiagnostics` task boundaries.
- `python -m qualification.diagnostics.nsys_extract` extracts task and runtime
  intervals from an Nsight Systems SQLite export.
- `python -m qualification.diagnostics.nsys_validate` validates the expected
  CUDA allocation and synchronization invariants in that export.
