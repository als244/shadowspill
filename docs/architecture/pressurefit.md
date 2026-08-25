# PressureFit

PressureFit is ShadowSpill's framework-neutral memory-policy planner. Given an
ordered [`Program`](ir.md), required initial and final residency, memory and
transfer capacities, and an optional physical-admission contract, it chooses:

- which logical objects remain in execution memory between tasks;
- where objects leave execution memory by release or eviction;
- which task boundary triggers each later fetch;
- whether a clean release/fetch pair is coalesced; and
- the minimum-makespan valid candidate among the bounded policies it evaluates.

PressureFit does not capture graphs, inspect PyTorch tensors, partition a
model, construct graph pairs, execute numerical kernels, or advance the
runtime worker. It consumes only Program facts and machine parameters. A
Program with no alternatives is a normal input; training programs may supply
several legal task selections through the separate [recomputation
selector](recomputation-selection.md).

## Contract at a glance

| Item | Contract |
|---|---|
| Primary input | One immutable `Program` whose tasks are ordered and whose objects, aliases, profiles, and dependencies are valid. |
| Boundary conditions | `initial_residency` and `final_residency` tuples of `ResidencySpec` values. |
| Machine input | `SimulationConfig`: execution capacities, spill capacity, directional transfer bandwidths, and latencies. |
| Search input | `PressureFitOptions`: initial placement, residency strategies, fetch rules, coalescing, repair limit, capacity-refinement granularity, and worker count. |
| Optional physical input | `AdmissionFacts`: task allocation steps, output/replacement ownership, storage handoffs, pool capacity, and alignment. |
| Output | `PressureFitResult`: selected task alternative, `MemorySchedule`, full `SimulationResult`, and structured diagnostics. |
| Feasibility authority | The planner preflight, the simulator, and physical admission when an `AdmissionFacts` is supplied. |
| Optimization scope | Minimum simulated makespan among the finite candidates actually generated and repaired, not a global optimum over all possible schedules. |

`validate_schedule_feasibility()` performs a necessary-condition preflight on
the Program and its legal task selections. It does not validate an already
annotated `MemorySchedule`; `simulate()` is the independent schedule validator.

## Inputs

### Program facts

For each selected task, PressureFit uses:

- declared inputs, outputs, and mutations;
- the task's execution device and profiled runtime;
- profiled anonymous workspace bytes;
- dependencies and execution order.

For each logical alias bundle, it uses:

- byte size and execution device;
- initial and required final location;
- whether a current spill copy may be retained;
- production, access, and write boundaries.

Object names, semantic tensor roles, model families, and operator names do not
enter policy decisions. Zero-byte alias bundles remain semantic objects but do
not contribute physical pressure or transfer actions.

### Machine facts

For each execution device $d$, `SimulationConfig` supplies logical planning
capacity $C_d$, fetch and evict bandwidth, and route latency. It also supplies
the spill-pool capacity. The runtime calibrates transfer behavior; PressureFit
only consumes the resulting values.

Workspace is task-local. PressureFit subtracts task $i$'s workspace only at
the boundary where that task's inputs, fresh outputs, and workspace must
coexist. It does not subtract a global maximum workspace from every boundary.

When physical admission is enabled, `AdmissionFacts.pool_capacity_bytes`
is the complete execution-pool capacity, while
`AdmissionFacts.object_capacity_bytes` is the current logical capacity
offered to PressureFit. Physical refinement may reduce the latter without
changing the former.

### Search controls

The default `PressureFitOptions` evaluate four residency-strategy labels, four
fetch-trigger rules, and ordinary/coalesced emission: 32 candidate policies
per legal task-selection problem. `workers=0` evaluates independent problems
with up to the available logical CPUs.

`capacity_refinement_bytes` decides how much capacity a plan gives back when
its layout does not fit the pool, 256 MiB by default. Stepping costs rounds
and buys plan quality; zero hands back the whole shortfall and converges in
the fewest rounds. Capacity is a property of a
plan, not of the search — two candidates can answer at different capacities in
the same call — and it is described in
[physical admission](physical-admission.md).

Candidates place layouts and publish what they place to a shared record, so
which plans are worth measuring depends on what has already been placed. That
makes the search order-dependent by default; a caller that needs a reproducible
answer keeps a record per search rather than sharing one.

## Output

`PressureFitResult` preserves all inputs needed to explain or replay the
decision:

- `program`, `initial_residency`, `final_residency`, and `simulation_config`;
- selected `selections` for any Program alternatives;
- selected `schedule` with initial placement and ordered release, offload, and
  prefetch actions;
- full `simulation`, including makespan, task/transfer intervals, and peaks;
- `diagnostics`, including every problem and candidate outcome, repair counts,
  component work, schedule digests, and physical-refinement history;
- the original `admission_facts`, when supplied.

The Python API uses `prefetch`/`offload` as serialized action names. In
explanatory text, these are fetch and evict, respectively.

## Mathematical formulation

### Selected task sequence and boundaries

Let $r\in\mathcal R(P)$ be one legal task selection exposed by Program $P$.
Most Programs have a singleton set. For one $r$, let

\[
\mathcal T_r=(\tau_0,\ldots,\tau_{n-1})
\]

be the selected task sequence and

\[
\mathcal B=\{-1,0,\ldots,n-1\}
\]

its boundaries. Boundary $-1$ precedes task 0; boundary $i$ follows task $i$.
Thus task $i$ consumes its inputs from boundary $i-1$ and publishes its
outputs at boundary $i$.

Let:

- $\mathcal A$ be the alias bundles;
- $s_a$ and $d(a)$ be alias $a$'s bytes and execution device;
- $c_i$, $w_i$, and $d_i$ be task $i$'s profiled runtime, workspace,
  and execution device;
- $H_a\subseteq\mathcal B$ be the required residency anchors of $a$;
- $F_i\subseteq\mathcal A$ be task $i$'s fresh output aliases;
- $\lambda_a^0$ and $\lambda_a^f$ be declared initial and final
  locations.

The anchor set contains the pre-task boundary for every input or mutation,
the post-task boundary for every output or mutation, an initial device
boundary when required, and the final boundary when final device residency is
required.

### Residency pressure

A residency plan $R$ maps each alias $a$ to an ordered tuple of disjoint
inclusive spans whose covered boundaries $R_a\subseteq\mathcal B$ contain
every anchor:

\[
H_a\subseteq R_a.
\]

Let $\chi_{a,b}(R)$ be one when alias $a$ is physically charged at boundary
$b$. It follows span membership with one task-boundary refinement used by the
implementation: an alias whose last access has completed and whose final
location is not execution memory may depart before the next task is admitted.

The object capacity available at boundary $b$ is

\[
\widehat C_{d,b}=
\begin{cases}
C_d-w_{b+1}, & b+1<n\ \text{and}\ d_{b+1}=d,\\
C_d, & \text{otherwise}.
\end{cases}
\]

Fresh outputs must have capacity before their producing task starts, even
though their residency anchor is the post-task boundary. Define

\[
Q_{d,b}(R)=
\sum_{a\in F_{b+1}:d(a)=d}
s_a\,\mathbf 1[\chi_{a,b}(R)=0]
\]

for $b+1<n$, and zero at the terminal boundary. The analytic pressure
constraint is

\[
M_{d,b}(R)=
Q_{d,b}(R)+
\sum_{a:d(a)=d}s_a\chi_{a,b}(R)
\le \widehat C_{d,b}
\qquad \forall d,b.
\]

This boundary model is necessary but not sufficient. It deliberately omits
exact FIFO timing, simultaneous transfer reservations, physical range
fragmentation, and causal range reuse. Those constraints are enforced by
physical admission and simulation.

### Legal residency reductions

The required seed for alias $a$ is the inclusive hull of its anchors:

\[
R_a^0=\operatorname{Hull}(H_a).
\]

Greedy initial placement may additionally extend selected spill-origin aliases
to boundary $-1$ when capacity permits and doing so is likely to hide an
early fetch. A legal cut removes an anchor-free subinterval from one current
span, possibly splitting it in two. Therefore every generated plan satisfies

\[
H_a\subseteq R_a\subseteq R_a^0
\]

apart from an explicitly chosen greedy initial extension.

A legal cut normally removes an anchor-free run. It may also insert a span
break immediately after an anchor when that anchor has no later access tied to
the same boundary. In that second case the covered-boundary set is unchanged,
but the span segmentation records a legal departure/re-entry point. A gap or
break between two spans creates a departure and, if the later span was not
produced there, a fetch window. A current spill copy with no intervening write
allows a release; otherwise the departure is an eviction. Final residency
determines whether the last span is retained, released, or evicted.

### Candidate space and objective

Let $\mathcal S$ be the configured residency strategies,
$\mathcal F$ the configured fetch rules, and
$\mathcal K\subseteq\{0,1\}$ the enabled coalescing modes. For one legal task
selection $r$, one policy tuple is

\[
\theta=(s,f,k)\in\mathcal S\times\mathcal F\times\mathcal K.
\]

`Reduce` greedily constructs an analytically fitting residency plan,
`Emit` converts its gaps to a schedule, and bounded `Repair` monotonically
changes fetch constraints or adds measured boundary pressure after admission
or simulation rejects a candidate:

\[
\Gamma_{r,\theta}=
\operatorname{Repair}
\left(
\operatorname{Emit}
\left(
\operatorname{Reduce}(R_r^0,s),f,k
\right)
\right).
\]

Let `Admit` be true when the optional physical topology accepts a schedule,
and let

\[
\operatorname{Simulate}(P,r,\Gamma;\mathcal H)=(v,m)
\]

return validity and makespan under the full machine model. PressureFit selects

\[
(r^\star,\theta^\star)=
\arg\min_{r\in\mathcal R(P),\theta}
\left(m_{r,\theta},\operatorname{ordinal}(r,\theta)\right)
\]

over candidates for which analytic reduction succeeds, admission succeeds
when present, and $v=\mathrm{true}$. The ordinal is a deterministic
tie-breaker. This is an optimum over the bounded generated family, not over
all possible residency intervals, alternative selections, or transfer
triggers.

## Current algorithm

### 1. Necessary-condition preflight

For every legal task-selection problem, the planner derives anchors, fresh-output
reservations, and per-boundary capacity. At least one problem must fit its
required anchor/output floor. This catches an individual task whose required
inputs, outputs, and workspace cannot coexist before candidate search.

### 2. Build one problem per legal task selection

PressureFit obtains the finite set of legal selections from the Program. The
training-specific policy used to construct this set is documented separately
in [Recomputation selection](recomputation-selection.md). Each problem is
projected into indexed task, alias, simulation, and optional admission arrays.

### 3. Seed residency

`InitialPlacement.REQUIRED` uses only the anchor hull. The default
`InitialPlacement.GREEDY` also considers spill-origin aliases first consumed
after task 0. It orders them deterministically using first-use time, estimated
fetch deadline miss, transfer cost, size, and alias order, then preplaces each
one that fits initial capacity.

### 4. Reduce analytic pressure

For one residency strategy:

1. Find the boundary/device pair with the largest byte excess; break ties by
   earlier boundary and stable device priority.
2. Enumerate legal cuts that relieve that boundary without removing anchors.
3. Rank cuts using the strategy's lower-is-better score.
4. Apply the best cut and update only that alias's pressure contribution.
5. Repeat until every analytic boundary fits or no legal cut exists.

The stall score first minimizes estimated exposed

\[
\max(t_{\mathrm{depart}}+t_{\mathrm{evict}}+t_{\mathrm{fetch}}
-t_{\mathrm{deadline}},0).
\]

The transfer score omits this first term. Both then prefer no required
write-back, removable greedy initial placement, later first use, larger
objects, longer removable spans, and stable alias/boundary identity.

### 5. Emit actions and choose fetch triggers

Residency gaps determine whether an alias is released or evicted and the legal
window for its next fetch. The selected fetch rule picks exactly one task
boundary in that window. Emission sorts actions by task, then release, evict,
fetch, then alias identity.

Coalescing removes a clean release and fetch for the same alias at the same
task boundary. Dirty values require eviction and are never removed by this
coalescing rule.

### 6. Admit, simulate, and repair

When an admission topology is supplied, the planner dry-runs the
task allocation path, output ownership, transfer reservations, retirements,
and causal reuse through the production memory-pool policy. It emits physical
deltas and reuse dependencies consumed by simulation.

For a repairable admission failure, the candidate tries, in order:

1. advancing a fetch to a compatible release boundary;
2. delaying a fetch toward its consumer;
3. adding the measured physical deficit to the failing analytic boundary and
   rerunning residency reduction.

For a repairable simulator capacity failure, it first delays an implicated
fetch and then adds simulator-observed boundary pressure. Every change is
monotonic and contributes to `max_repair_attempts`. A non-capacity
contradiction is rejected directly.

Simulator capacity failures are now rare, because a prefetch or task launch
with nowhere to go waits for room rather than ending the simulation. A plan
that comes up short is slower, not rejected, and it reaches selection with a
real makespan and a `device-capacity` stall recording what it waited for.
What still fails is a plan over budget before it starts, an offload with no
room in the spill pool, and a plan that can never make room, which deadlocks.

One consequence is worth stating plainly: because repair runs only while the
simulation fails, a plan that succeeds with capacity stall is accepted
without being repaired. Reducing that stall is an optimization the search
does not currently attempt.

### 7. Select and materialize

Each valid candidate has an exact simulated makespan and schedule digest. The
planner selects the lowest `(makespan, candidate ordinal)` pair across all
problems, decodes that one indexed schedule, evaluates its physical admission
once more, and materializes the full `SimulationResult`.

If all otherwise logical candidates fail physical admission, the outer
orchestrator reduces logical object capacity and repeats the complete search.
The minimum capacity decrements are 128, 256, 512, and 1,024 MiB, then 1,536,
2,048 MiB, and so on in 512 MiB growth steps. Each decrement is at least the
reported contiguous deficit rounded to 2 MiB. The physical pool capacity does
not change, and every refinement is retained in diagnostics.

## Pseudocode

```text
PressureFit(program, initial, final, machine, options, admission):
    require the planner and simulator ABIs
    graph pairs = legal_task_selections(program)
    require some selection's anchor/output floor to fit

    object_capacity = machine.device_capacity
    refinements = []

    loop:
        problems = compile_indexed_problems(
            program, graph pairs, initial, final,
            machine.with_capacity(object_capacity), admission
        )

        outcomes = evaluate problems independently:
            seed = required_anchor_hulls(problem)
            if options.initial_placement == GREEDY:
                seed = preplace_fitting_spill_objects(seed)

            for strategy in options.residency_strategies:
                base = reduce_until_analytic_capacity_fits(seed, strategy)

                for fetch_rule in options.prefetch_rules:
                    for coalesced in enabled_coalescing_modes:
                        residency = maybe_extend_interval_entries(
                            base, fetch_rule
                        )
                        constraints = empty

                        for attempt in 0..max_repair_attempts:
                            schedule = emit_actions(
                                residency, fetch_rule, coalesced, constraints
                            )

                            physical = admit_if_configured(schedule)
                            if physical is repairably infeasible:
                                if fetch can be advanced or delayed:
                                    add monotonic fetch constraint
                                    continue
                                add physical deficit at failing boundary
                                residency = reduce_again(seed, strategy)
                                continue
                            if physical is infeasible:
                                reject candidate

                            simulation = simulate(schedule, physical)
                            if simulation is valid:
                                record candidate and makespan
                                break
                            if simulation is repairable:
                                delay fetch or add boundary pressure
                                reduce again when pressure changed
                                continue
                            reject candidate

        if any candidate is valid:
            winner = minimum(outcomes, key=(makespan, stable_ordinal))
            return materialize_result(winner, refinements)

        if failures prove a physical contiguous deficit:
            decrement = max(round_up_2MiB(deficit), scheduled_refinement())
            object_capacity -= decrement
            append refinement diagnostics
            continue

        raise infeasible or search-exhausted error
```

## Built-in candidate policies

The three candidate axes have separate responsibilities:

1. The residency strategy decides which optional object-boundary cells to
   remove and therefore which residency gaps exist.
2. The fetch-trigger rule decides when each gap's already-required return is
   enqueued; it does not decide whether the gap exists.
3. Coalescing optionally removes a clean release/fetch pair at one boundary.

No row in the tables below is assumed to dominate another. PressureFit emits
the resulting schedule, applies physical admission when configured, and lets
the simulator measure the combined compute, lane, readiness, and capacity
effect.

### Residency strategies

| Strategy | Cut score | Early-fetch headroom | Effect on PressureFit behavior |
|---|---|---:|---|
| `headroom-stall` | Minimize estimated exposed stall first | Yes | Charges each fetched span one boundary early while reducing pressure. This may create more or different gaps so early destination reservations fit, trading object residency or transfer traffic for fewer admission conflicts and less exposed readiness stall. |
| `headroom-transfer` | Prefer cuts that avoid write-back before other tie-breaks | Yes | Uses the same conservative early-fetch charge, but removes the stall estimate from the first score position. It tends to favor clean releases and lower eviction work even when another cut has a better local overlap estimate. |
| `tight-stall` | Minimize estimated exposed stall first | No | Fits only the current logical residency/output pressure. It may retain more useful residency than a headroom candidate, but later trigger-time fetch reservations can expose pressure that admission or simulation must repair. |
| `tight-transfer` | Prefer cuts that avoid write-back before other tie-breaks | No | Combines tight logical accounting with the transfer-oriented cut order. It can reduce eviction traffic, while accepting greater risk that selected gaps or fetch timing expose stall or trigger-time capacity pressure. |
| `relaxed-stall` | Minimize estimated exposed stall first | No | Currently maps to the same reduction controls as `tight-stall`, so it produces no distinct pressure behavior unless another implementation control is added. It remains a separate candidate identity in diagnostics. |

Headroom accounting charges a fetched residency span one boundary earlier
than its logical entry. It is conservative boundary accounting, not a transfer
start prediction. The transfer lane and exact destination lifetime remain the
simulator and admission model's responsibility.

### Fetch-trigger rules

| Rule | Mechanical behavior | Expected pressure/latency tradeoff |
|---|---|---|
| `packed-fifo` | Work backward from consumer deadlines while packing each device's single fetch lane; earlier fetches account for residual occupancy left by later packed work. | Seeks lane utilization and overlap across the complete reload set. It may enqueue a destination earlier than capacity permits, so admission can delay it or force another residency cut. |
| `packed-fit` | Start with packed FIFO triggers, then move implicated triggers later until their early destination occupancy fits analytic capacity where possible. | Reduces trigger-time capacity pressure relative to unconstrained packing, at the cost of less transfer lead time and potentially more consumer stall. |
| `interval-entry` | Extend each later residency span toward earlier boundaries while exact analytic capacity fits, then place fetches with packed FIFO. | Uses otherwise idle object capacity to create more transfer lead time. It can hide latency, but increases how long fetched destinations occupy execution memory. |
| `latest-safe` | Independently subtract each fetch's duration from its ideal consumer deadline and choose the latest task boundary no later than that time. | Limits early residency for each object, but does not jointly pack the FIFO. Several individually safe choices may queue behind one another and expose lane-induced consumer stall. |
| `demand` | Trigger at the final legal enqueue boundary, normally immediately before the first consumer. | Minimizes pre-consumer destination occupancy and is often capacity-friendly, but deliberately exposes most or all fetch latency when the object is not already ready. |

These rules choose enqueue boundaries, not wire start timestamps. The fetch
lane remains FIFO, and destination capacity is reserved at the trigger even if
earlier lane work delays the copy.

### Coalescing mode

| Mode | Mechanical behavior | Impact |
|---|---|---|
| Ordinary | Preserve every release, eviction, and fetch implied by the residency spans. | Represents the transition plan literally and may expose a redundant clean release/fetch pair at one boundary. |
| Coalesced | Remove a release and fetch for the same clean alias at the same trigger boundary. Eviction/fetch pairs remain. | Retains the already-current execution copy across that boundary, avoiding needless queue work and transfer latency without hiding a required write-back. |

## Guarantees and limits

PressureFit guarantees:

- anchor-preserving residency plans;
- deterministic candidate identity, ordering, and tie-breaking;
- exact task-local workspace accounting in analytic boundary capacity;
- a simulator-valid returned schedule;
- physical-admission validity when an admission topology is supplied;
- structured evidence for candidate work, repairs, rejection, and selection.

It does not guarantee:

- a global minimum over arbitrary residency intervals or transfer triggers;
- exhaustive selection of large Program-alternative products;
- feasibility after runtime behavior violates the admitted allocation
  contract.

`PressureFitInfeasibleError` means a necessary-condition preflight failed or
every candidate in the bounded evaluated graph pairs was rejected without a
remaining repair path. It does not prove that no schedule outside that
graph pairs exists. `PressureFitSearchExhaustedError` means at least one
repairable path reached its configured repair ceiling, so even graph pairs-level
infeasibility was not established.

## Implementation map

| Layer | Responsibility |
|---|---|
| `shadowspill.planner.pressurefit()` | Input validation, problem concurrency, and winner materialization. |
| `csrc/src/planner/residency.c` | Indexed anchor geometry, pressure accounting, legal cuts, scoring, and reduction. |
| `csrc/src/planner/schedule.c` | Gap transitions, fetch-window placement, action emission, and trigger constraints. |
| `csrc/src/planner/graph pairs.c` | Candidate loop, caches, admission/simulation repair, selection, and work diagnostics. |
| `csrc/src/planner/admission.c` | Physical allocation and causal-reuse admission. |
| `shadowspill.simulator` / `csrc/simulator` | Independent schedule replay and makespan authority. |

The production path requires the planner and simulator. Readable
Python implementations live only under `reference/python/pressurefit` and
`reference/python/simulator`; they are differential-test oracles and do not
silently replace a missing or ABI-incompatible library.

Previous: [Recomputation selection](recomputation-selection.md). Next:
[Physical admission and offset handling](physical-admission.md).
