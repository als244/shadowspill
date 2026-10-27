# SSD numerical qualification

`numerical_ssd` uses the numerical matrix's cases, compiled references, tolerances,
checkpoint replay and physical-budget checks. Only the planned arm's spill
backend changes. It uses the same build/profile/plan/fresh-worker pipeline as
pinned-host execution; references do not depend on the spill backend.

```bash
python -m qualification.gates numerical_ssd --config gates.json --keep-going
```

```json
{"numerical_ssd": ["--ssd-directory", "/local/ssd", "--ssd-staging-mib", "256"]}
```

Use an existing local SSD directory with space for the 32 GiB spill capacity.
`SHADOWSPILL_SSD_DIRECTORY` supplies the default directory; an explicit option
wins. `--ssd-chunk-mib` and `--ssd-queue-depth` tune the bounded staging pipeline.

The saved settings retain the SSD configuration. Communication measurement
calibrates that backend with bounded probes; Executor opens its own temporary
spill file and closes it after the case. Build and task profiling do not open the
SSD backend. Initialization/checkpoint export use the ordinary bounded tensor
interfaces. Results, Plans and reference checkpoints remain separate from the
spill file and are retained.

This runner's saved-phase wiring has a passing tiny training/checkpoint probe.
The full-size SSD matrix remains a separate target run from the default
pinned-host gate. See the [SSD API](../../docs/python/api/ssd.md) for pool details.
