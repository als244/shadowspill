# Build, profile, plan and execute

Use separate preparation and execution processes. Build produces executable
tasks; profiling supplies timings and allocation evidence; CPU planning selects
a schedule; a fresh worker executes it. The [complete training example](../examples/generic-training.md)
defines all application callbacks used below.

| Phase | Requires a device? | Saved result |
| --- | --- | --- |
| Build | Representative GPU; model-owned collectives also need their participants | `Program` |
| Profile compute | Representative GPU | `Profiles`, then `ProfiledProgram` |
| Measure communication | Target GPU(s) and spill/collective configuration | `CommunicationInfo`; optional overlap profiles |
| Plan and report | CPU only | `Plan`, report |
| Execute | Target worker(s), fresh allocator state | Training results and optional checkpoints |

## Build and profile

```python
import shadowspill as ss

program = ss.build_step(
    model, optimizer=optimizer, objective=objective,
    example_args=(inputs, targets), num_examples=4,
)
program.save("program.ssp")
profiles = ss.measure_program(program)
profiles.save("profiles.ssp")
ss.apply_profiles(program, profiles).save("profiled.ssp")
ss.measure_communication(program=program).save("communication.ssp")
```

The application defines model, objective, optimizer and representative inputs.
`num_examples` fixes the rank-local calls in one update; `example_calls` can
describe calls with different shapes. `build_forward` captures a forward-only
computation; `build_forward_backward` omits optimizer updates.

Build accepts CPU/meta model state and finalizes selected task implementations
on the device. It is not a CPU-only preparation phase. Neither build nor compute
profiling creates execution/spill pools.

Separate controls determine checkpoint stages (`partition`), execution regions
and axis sizes (`chunking`), and saved intermediates (`graph_pair_policy`).
These change executables and require build/profile before planning. See the
[partitioning example](../examples/custom-partitioning.md) and [policy reference](api/frontend.md).

[Build details](api/build.md) explain capture, finalization and saved code.
[Profiling](api/profiling.md) documents conditioning, sample limits, allocation
observations and reusable task measurements. [Communication](api/communication.md)
covers transfer and overlap calibration; [distributed training](api/distributed.md)
shows the multi-worker workflow. Optional [pool-memory calibration](api/pool-memory.md)
measures device costs of large spill pools separately.

## Plan on CPU

```python
import shadowspill as ss

plan = ss.plan_program(
    ss.ProfiledProgram.load("profiled.ssp"),
    memory_budgets=ss.MemoryBudgets(execution_bytes=40 << 30, spill_bytes=128 << 30),
    communication_info=ss.CommunicationInfo.load("communication.ssp"),
)
plan.save("training.ssp")
ss.write_report(plan, "report")
```

Planning and reporting do not import PyTorch or initialize a device. Reuse the
same measured artifacts for different budgets, traversal orders and planner
settings. `sweep_programs` compares already-profiled Programs; it does not build
candidates. [Quickstart](../../benchmarking/quickstart.md) and the
[shared recipe runner](../../workloads/recipes/README.md) can also construct and
profile candidate microbatch sizes, chunk sizes and retention policies.

A numeric workspace estimate permits simulation, but execution requires observed
allocation evidence for the saved implementations. Check
`plan.simulation_only_reasons`. Missing memory is never treated as zero. Physical
placement is necessary for execution; worker startup still checks compatibility
and actual available memory.

## Execute in a fresh worker

```python
import shadowspill as ss

with ss.Trainer(ss.Plan.load("training.ssp")) as trainer:
    trainer.initialize(fill_tensor)
    trainer.fit(training_batches, steps=100)
```

`fill_tensor` fills bounded destination views from checkpoint data or random
initialization. Training batches supply the Plan's fixed call structure and
example count. Trainer does not build, profile or plan. Construct it before
another device allocator initializes in this process.

See [Trainer](api/training.md) for inputs, schedules, evaluation and checkpoints,
or [Executor](api/executor.md) for asynchronous invocation and shared Plan state.
The default pinned-host pool uses the admitted spill peak plus 5%, capped at the
planned spill budget. [Pool-memory rules](api/pool-memory.md) describe calibrated
capacity and portability.

## Reuse and inspect

[Artifact reference](api/artifacts.md) describes save/load and compatibility.
[Artifact and cache guidance](artifact-store.md) distinguishes durable artifacts
from disposable compiler caches. [Plan reports](plan-report.md),
[execution diagnostics](step-diagnostics.md) and [failure handling](failures.md)
explain how to inspect outcomes.
