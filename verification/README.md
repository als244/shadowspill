# Verification launchers

Run the complete Llama 3, Qwen 3.5, and OLMoE correctness matrix for both the
pure-PyTorch and external mlops implementations:

```bash
python -m verification.run_model_correctness \
  --build-dir /tmp/shadowspill-build-full \
  --cache-dir qualification/results/cache
```

Every case runs a whole-objective `torch.compile(fullgraph=True)` reference on
PyTorch's standard allocator and ShadowSpill in separate processes. The matrix
requires five two-microbatch optimizer steps, compiled-reference numerical parity,
step-three checkpoint/two-step bitwise replay, real evict/fetch transfers, selected
recomputation, and measured physical-budget enforcement. Individual evidence
and `summary.json` are written below `qualification/results/numerical_matrix`.

Select a subset or override a budget without editing the script:

```bash
python -m verification.run_model_correctness \
  --build-dir /tmp/shadowspill-build-full \
  --models qwen35 olmoe \
  --implementations pytorch mlops \
  --budget qwen35=10GiB \
  --budget olmoe=10GiB \
  --reuse-reference \
  --keep-going
```

`--reuse-reference` is opt-in because a newly generated compiled reference is
safer after model, optimizer, PyTorch, or compiler changes. The per-case implementation remains in
`qualification.numerical.run`; this launcher only orchestrates the matrix and
summarizes its artifacts.

Built-in model configuration and microbatch geometry can be supplied inline as
JSON or loaded from a file by prefixing its path with `@`:

```bash
python -m verification.run_model_correctness \
  --build-dir /tmp/shadowspill-build-full \
  --models llama3 \
  --implementations pytorch mlops \
  --budget llama3=12GiB \
  --model-config '{"n_layers": 16, "max_seq_len": 256}' \
  --data-geometry '[
    {"token_shape": [1, 128], "sequence_lengths": [31, 47, 50]},
    {"token_shape": [2, 96], "sequence_lengths": [64, 64, 64]}
  ]'
```

For a model outside the built-in registry, provide a stable model name, an
explicit physical budget, and a factory:

```bash
python -m verification.run_model_correctness \
  --build-dir /tmp/shadowspill-build-full \
  --models diffusion_transformer \
  --implementations pytorch \
  --budget diffusion_transformer=24GiB \
  --case-factory my_project.shadowspill_cases:build_case \
  --model-config @configs/dit.json \
  --data-geometry @configs/dit_microbatches.json \
  --case-option objective='"flow_matching"' \
  --case-option optimizer='{"name":"AdamW","lr":0.0001}'
```

The factory is called independently in compiled-reference and planned processes with the
keyword arguments `model_name`, `model_implementation`, `seed`, `model_config`,
`data_geometry`, and `case_options`. It returns an object exposing `model`,
`microbatches`, `objective(model, *microbatch)`, `optimizer(parameters)`, and an
`implementations()` context manager, plus matching `family` and
`model_implementation` fields. This keeps model-, data-, objective-, optimizer-,
and custom-operation policy outside the verification launcher.

The compiled-reference artifact records a digest of the complete request,
including the reference execution mode. `--reuse-reference`
is rejected if the model name, implementation, seed, model config, geometry,
factory, or case options differ.

Every accepted planned run writes a `<case>_pressurefit/` directory containing
the exact `pressurefit()` input/output fixtures. Record a direct, cache-free
Python baseline with:

```bash
python -m verification.benchmark_pressurefit \
  qualification/results/numerical_matrix/pytorch_llama3_pressurefit \
  qualification/results/numerical_matrix/mlops_llama3_pressurefit \
  --repeats 3 \
  --output qualification/results/pressurefit_benchmark.json
```

The benchmark bypasses the selection cache, times the public PressureFit
implementation, and fails unless schedule, recomputation selections, complete
simulator result, and candidate diagnostics reproduce the recorded expected
digest. After replacing the implementation, pass the saved report back with
`--baseline qualification/results/pressurefit_benchmark.json`; matching fixture
suites then report direct `pressurefit()` speedup. No PyTorch capture,
compilation, profiling, materialization, admission, or outer `plan_step()` work is
inside the timed interval.
