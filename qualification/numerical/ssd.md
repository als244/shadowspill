# SSD numerical qualification status

The `numerical_ssd` gate and its configuration are retained, but the saved-phase
runner currently rejects non-host spill in `numerical/prepare.py`. It must be
wired through communication calibration and Executor before this gate can run.
Core SSD saved execution has separate tests; those do not qualify this matrix.

The intended comparison uses the same numerical cases, references, tolerances,
checkpoint replay and physical-budget checks. Only the planned arm's spill
backend changes. References must not be regenerated merely to select SSD storage.
The retained command-line settings are `--spill-pool ssd`, `--ssd-directory`,
`--ssd-staging-mib`, `--ssd-chunk-mib` and `--ssd-queue-depth`.

The migration should save the pool configuration, calibrate that backend, and
pass it to Executor. Model build and task measurement must stay independent of
spill storage. Reuse the existing bounded initialization/checkpoint paths; do not
restore the retired model-based runner or add another execution path.

See the [SSD API](../../docs/python/api/ssd.md) for supported pool configuration
and staging behavior.
