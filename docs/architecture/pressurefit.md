# PressureFit

PressureFit is the [search](search.md) ShadowSpill ships: one implementation
of `SearchAlgorithm`, and the only one today. Everything the planner asks of
any search is on that page, and the contract an implementer writes against is
in [search algorithm](search-algorithm.md); this one is about how PressureFit
answers.

Given a [program](program.md), required initial and final residency, memory
and transfer capacities, and an optional physical-admission contract, it
chooses:

- which logical objects remain in execution memory between tasks;
- where objects leave execution memory by release or eviction;
- which task boundary triggers each later fetch;
- whether a clean release/fetch pair is coalesced; and
- the minimum-makespan valid candidate among the bounded policies it
  evaluates.

PressureFit does not capture graphs, inspect PyTorch tensors, partition a
model, construct graph pairs, execute numerical kernels, or advance the
runtime worker. It consumes only program facts and machine parameters.

## Resolved programs, and who expands them

A program arrives with its [task alternatives](program.md#task-alternatives)
still open. Fixing one option for every group gives a **resolved program**: a
concrete task set. A program with no groups is already resolved.

Expanding a program into resolved programs is PressureFit's own work, not the
planner's, because how many ways of fixing the alternatives are worth
planning -- and in what order -- is a judgement about the search. PressureFit
plans several and answers with the best across them, and the order is part of
the algorithm: a plan admitted under any resolved program bounds the search
under every later one. Which shares to expand is
`PressureFitOptions.resolution_options`. See [ordering the
resolutions](#trajectories).

Program and problem are not the same word here: the program is what the
caller has, and the [problem](planning-problem.md) is the planning question
derived from it -- residency, boundaries, capacities. One compiled problem is
one resolved program.

"PressureFit" names this search and nothing else. What surrounds it --
turning a budget into a machine, keying the answer, holding it to a plan
already in hand, admitting the winner -- belongs to `plan_program()` and is
the same whichever search runs; see [plan search](search.md) and [planning
orchestration](planning.md). The entry to the search itself is
`shadowspill.planner.search.algorithms.pressurefit`.

## Contract at a glance

| Item | Contract |
|---|---|
| Primary input | One immutable `ShadowSpillProgram`, whose tasks are ordered and whose objects, aliases, profiles, and dependencies are valid. Its alternatives are expanded here into resolved programs, in the order they are searched. |
| Boundary conditions | `initial_residency` and `final_residency` tuples of `ResidencySpec` values. |
| Machine input | `SimulationConfig`: execution capacities, spill capacity, directional transfer bandwidths, and latencies. |
| Planner input | `SearchOptions`: worker count, determinism, and the evict-eligibility floor. What every search is told. |
| Search input | `PressureFitOptions`: resolution options, initial placement, residency strategies, fetch rules, coalescing, repair limit, capacity-refinement granularity, and write-back splitting. PressureFit's own, opaque to the planner. |
| Optional physical input | `AdmissionFacts`: task allocation steps, output/replacement ownership, storage handoffs, pool capacity, and alignment. |
| Output | `ProgramPlanResult`: selected task alternative, `MemorySchedule`, full `SimulationResult`, and structured diagnostics. |
| Feasibility authority | The planner preflight, the simulator, and physical admission when an `AdmissionFacts` is supplied. |
| Optimization scope | Minimum simulated makespan among the finite candidates actually generated and repaired, not a global optimum over all possible schedules. |

`validate_schedule_feasibility()` performs a necessary-condition preflight on
the program and its legal task selections. It does not validate an already
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

The default `SearchOptions` evaluate two residency-strategy labels
(`headroom-stall` and `tight-stall`), four fetch-trigger rules, and
ordinary/coalesced emission: 16 candidate policies per resolved program. The
two transfer strategies stay available and off by default: a
transfer strategy differs from its stall twin only in how it accounts for
transfers already in flight, and the plans the two reach are the same far
more often than not, so the default buys half the search for the rare small
win. Coalescing stays in: joining adjacent transfers changes what the
simulator sees, and the coalesced twin does win on its own.

`split_write_backs` lets a plan that has simulated split an eviction that
held something up, described under
[split](#split-clean-early-release-late). It widens what the search may
consider rather than deciding anything: the split plan is simulated and kept
only if it is faster and still places.

`capacity_refinement_bytes` decides how much capacity a plan gives back when
its layout does not fit the pool, 256 MiB by default. Stepping costs rounds
and buys plan quality; zero hands back the whole shortfall and converges in
the fewest rounds. Capacity is a property of a
plan, not of the search — two candidates can answer at different capacities in
the same call — and it is described in
[physical admission](physical-admission.md).

`record_reduction_steps` turns on the per-candidate trajectory described
under [Trajectories](#trajectories). It changes nothing about the search.

`minimum_object_bytes_evict_eligible` takes objects out of the search: one
smaller than it is never cut, so it is never evicted and fetched mid-step.
Every lease of such an object gets a static home in the
[resident slice](fixed-placement.md#the-resident-slice), whose size is known
at preparation and taken out of the capacity given to the reducer. The
default is 1 MiB, the size below which a copy is latency-bound and its bytes
hardly relieve a boundary; zero exempts nothing, which is what a caller
planning byte-sized objects wants.

Candidates place layouts and publish what they place to a shared record, so
which plans are worth measuring depends on what has already been placed. That
makes the search order-dependent by default. Scoping a record per search does
not remove that: one record is shared by every resolved program dispatched
concurrently within a call.

`deterministic` is how a caller gets a reproducible answer without giving up
its workers. The placement gate then consults only the candidate's own placed
plans instead of the shared record, so every outcome is a pure function of
that candidate's inputs and any worker count answers the same. It costs wall
time, because the shared bound is what lets a candidate skip measuring a plan
that cannot win. `workers=1` is reproducible too, and slower still.

A problem may carry the plan to beat: a plan for that resolved program
already in hand, found at a smaller capacity, say. The search measures it at
this capacity before any candidate runs — simulated, admitted, placed against
the pool — and answers with it unless a candidate does strictly better, so a
search handed one never answers worse than it. A plan that fits in less
memory fits in more, which is what lets a budget sweep hand each budget the
best plan found below it and plan monotonically in memory. In the default
mode the plan also seeds the shared record, so every candidate measures
against it from the start; in deterministic mode it changes only the answer.

### Workers and the unit of work

The unit of work is one **(resolved program, candidate) pair**. A worker takes
the next pair, evaluates it to completion, and takes another, so a worker that
draws a cheap candidate never waits on the problem it came from.

That granularity is what makes worker count and problem count independent.
`workers` sizes the threads whether the call was given one resolved program or
five: eight workers means eight threads either way. The threads belong to the
call, so two callers planning at once get their own and do not contend.
`workers=0` takes one per logical CPU; `workers=1` evaluates every pair on the
calling thread.

Ordering is the caller's: problems are searched in the order they are passed,
and the array order *is* the policy. Putting them in one call is also what
shares the placement record between them, so a plan placed under any resolved
program bounds the search under every other — which is the pruning that makes
searching several together cheaper than searching each alone.

Worker count is scheduling, not an input to the search: it changes neither
which plans are legal nor how they simulate. It does change how much of the
search is skipped, because a candidate is skipped when the record already
holds something it cannot beat. Per-candidate counters such as
`placements_attempted` therefore move with worker count, and so can the choice
between plans that tie.

Because workers interleave, a problem's `work.sections` is the sum of what its
candidates did rather than the time the call took; `started_ns`/`finished_ns`
are the elapsed-time counterpart, and [Output](#output) describes both.

## Output

`ProgramPlanResult` preserves all inputs needed to explain or replay the
decision:

- `program`, `initial_residency`, `final_residency`, and `simulation_config`;
- selected `selections` for any program alternatives;
- selected `schedule` with initial placement and ordered release, offload, and
  prefetch actions;
- full `simulation`, including makespan, task/transfer intervals, and peaks;
- `diagnostics`, including every problem and candidate outcome, repair counts,
  the work each did and the sections its time went to, when each ran
  (`started_ns`/`finished_ns`, nanoseconds from the start of the call, so
  candidates that overlap ran at the same time), schedule digests, capacity
  refinements, and — when asked for — each candidate's reduction trajectory;
- the original `admission_facts`, when supplied.

The Python API uses `fetch`/`evict` as serialized action names. In
explanatory text, these are fetch and evict, respectively.

## Mathematical formulation

### Selected task sequence and boundaries

Let $r\in\mathcal R(P)$ be one legal task selection exposed by program $P$.
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

Almost all of PressureFit is one cycle, repeated. A candidate reduces
residency until the analytic pressure fits, emits a schedule, simulates it,
and then either keeps what it got or repairs it and goes again. Everything
else — preparing the problem, setting up the workspace, adopting a winner —
happens once, around that cycle.

The sections below are that structure, and the names are load-bearing: they
are the same names the diagnostics report, the plan JSON carries, and
`PlanningSectionTiming` measures — `ShadowSpillPressureFitSectionTiming` across
the C ABI. A section is a disjoint span of work opened and closed by the
function that orchestrates it, so the time they account for sums exactly to
the time the step took. Reading a plan's timing and reading this page are the
same activity.

```text
per resolved program:  prepare -> setup -> [ per strategy ] -> select -> teardown
per strategy:          reduce  -> [ per fetch rule x coalescing mode ]
per candidate:         ( emit -> simulate -> repair
                              -> digest -> place -> settle )*
```

### Before the cycle: preflight and problem construction

For every resolved program, the planner derives anchors, fresh-output
reservations, and per-boundary capacity. At least one problem must fit its
required anchor/output floor. This catches an individual task
whose required inputs, outputs, and workspace cannot coexist, before any
candidate search happens. A resolved program that fails this derivation is
reported as infeasible on its own result; the others are evaluated together
as if it were absent, so a resolution that cannot fit at a capacity never
silences the ones that can, and a program whose every resolution fails it
is infeasible at that capacity.

The resolved programs are expanded here, from the alternative groups the
program carries: the finite set of legal selections, ordered
most-recomputed first. The training-specific policy that constructs the
alternatives is documented separately in [Graph-pair
selection](graph-pair-selection.md). Each selection becomes one
resolved program and one problem here. Deciding which resolved programs exist,
and in what order to try them, belongs above this search and never inside it.

### Prepare — deriving the residency problem

Projecting a resolved program into indexed task, alias, simulation, and
optional admission arrays. This is the only section that exists at the
problem level and not the candidate level: a candidate never prepares
anything, it inherits what preparation produced.

Seeding residency happens here too. `InitialPlacement.REQUIRED` uses only the
anchor hull. The default `InitialPlacement.GREEDY` also considers
spill-origin aliases first consumed after task 0, orders them
deterministically by first-use time, estimated fetch-deadline miss, transfer
cost, size, and alias order, and preplaces each one that fits initial
capacity.

The objects under `minimum_object_bytes_evict_eligible` are settled here as
well, because everything below assumes they are already gone. Each holds one
lease, from the trigger of its fetch — chosen once, as late as the ideal
timeline allows — or the start, to the boundary after its last access or the
end. Packing those leases on their own gives the resident slice, and the
device capacity the reducer sees loses it. From there these objects are absent
from the search: the floor and the pressure do not count them, the cut index
has no entries for them, greedy placement skips them, and the emitter issues
each fetch at the trigger the slice was sized for, so no fetch rule can move a
lifetime the slice was sized without. Every extent placement measures includes
the slice. Where the slice sits in the final layout, and why these objects get
static homes rather than a place in the main assignment, is
[fixed-offset placement](fixed-placement.md#the-resident-slice). A slice the
device cannot hold is a preflight failure of its own.

### Setup — schedule facts and the candidate workspace

Deriving the facts every candidate shares — access windows, transfer costs,
boundary capacities — and allocating the buffers the cycle reuses. Both are
paid once per resolved program rather than once per candidate, which is why
they are a section of their own rather than part of any candidate's time.

### Reduce — choosing what stays resident

Given a residency strategy, the reducer removes objects until the analytic
pressure fits every boundary:

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

Reductions are not cached. A repair trajectory cuts aliases monotonically,
so a candidate never returns to a residency it has already reduced; every
candidate pays its own reductions, the cost lands in `reduce_ns`, and
reductions a repair forces later are charged to the repair that forced
them, because that is what they cost. A reduction's fingerprint — a
128-bit hash of the packed residency — is what the later stages key on.

What a worker keeps is small and bounded. The last sixteen emitted
schedules are held in a ring keyed by residency fingerprint, fetch rule,
coalescing mode, and headroom; re-emission is recency-local, and an evicted
schedule is emitted again. Simulated outcomes are kept by the schedule's
digest with only their scalar results, so a schedule that recurs across
candidates is not simulated twice.

### Emit — turning residency gaps into an ordered schedule

Residency gaps determine whether an alias is released or evicted and the
legal window for its next fetch. The selected fetch rule picks exactly one
task boundary in that window. Emission sorts actions by task, then release,
evict, fetch, then alias identity, and finally applies whatever trigger
constraints earlier repairs recorded. A constraint that cannot be satisfied
ends the candidate here.

Coalescing removes a clean release and fetch for the same alias at the same
task boundary. Dirty values require eviction and are never removed by this
coalescing rule.

### Simulate — replaying the schedule for a makespan

The simulator replays the schedule against the machine facts and returns a
makespan, task and transfer intervals, and every place the plan came up
short of capacity and waited.

**Admit** is nested inside this section rather than beside it. When an
admission topology is supplied, the planner dry-runs the task allocation
path, output ownership, transfer reservations, retirements, and causal reuse
through the production memory-pool policy, and the physical deltas and reuse
dependencies it emits are what the simulation consumes. Because admission
runs as part of simulating, `admit_ns` is reported inside `simulate_ns` and
excluded from the disjoint sum.

A fetch or task launch with nowhere to go waits for room rather than
ending the simulation, as [simulation](simulation.md#trigger-time-capacity)
specifies. A plan that comes up short is therefore slower, not
rejected, and it reaches the rest of the cycle with a real makespan and a
`device-capacity` stall recording what it waited for. What still fails is a
plan over budget before it starts, an offload with no room in the spill pool,
and a plan that can never make room, which deadlocks.

### Repair — moving a transfer, or making room for one

For a repairable admission failure, the candidate tries, in order:

1. advancing a fetch to a compatible release boundary;
2. delaying a fetch toward its consumer;
3. adding the measured physical deficit to the failing analytic boundary and
   reducing again.

For a repairable simulator capacity failure it first delays an implicated
fetch, then adds simulator-observed boundary pressure and reduces again. A
plan that simulated but waited for memory is repaired the same way: the
shortfall it recorded stands in for an error, so a plan that merely stalls
takes the same path a plan that failed does. This is the difference that
matters most in practice — a plan that runs while waiting is valid but not
finished, and the waiting is time it pays.

A pressure repair asks the reducer for the shortfall the simulator measured.
When the next simulation comes up short at the same task and the same moment,
the room the reducer made did not become room where the simulator looks —
copies still in flight hold it, or the emitter packed the freed bytes again —
so a repeat asks for twice what the last round asked, up to the task's whole
request. Asking for the same bytes again would emit the same plan again, which
is how a candidate spends its whole repair budget at one task without moving.
An ask that no cut can meet is taken back for a plain ask, so a candidate is
only ever slower for having asked for more, never lost to it. A new capacity
round starts its count of repeats afresh.

Every change is monotonic and counts against `max_repair_attempts`, 256 by
default. A non-capacity contradiction is rejected directly. A move the
schedule already carries is not repeated, because repeating it would loop.

The default is generous on purpose: a candidate that runs out of repairs
answers with what it has, and the plans that need the most repairs are the
ones where memory is tightest, which are the plans most worth finishing.
It buys that with planning time, which is what the workers pay for.

Reductions this section triggers are measured inside it, so `repair_ns`
answers what the repair machinery actually costs rather than what its
bookkeeping costs.

### Split — clean early, release late

An eviction does two things at one boundary: it copies the object to spill and
it drops the device copy. The copy has to happen somewhere, but not
necessarily there.

An eviction only costs time when something is waiting for the room it frees,
and the room is not free until the copy has landed. A simulated plan says
exactly where that happened: every task and transfer interval carries the time
it was ready, the time it started, and a mask naming what it waited for. An
eviction whose copy overlaps a wait for device capacity is split in two: a
`WRITE_BACK` at the boundary where the object was last written, and a
`RELEASE` where the eviction was, which costs nothing because the spill copy
is already current by then.

The last write is where the copy goes because it is the earliest boundary at
which the copy is correct, and so the furthest from the boundary that was
waiting for it. Nothing here chooses a time on the lane. The simulator owns
the lane, and when several copies move at once it is the simulator that prices
the queue they form -- a pass that fitted copies into idle time it had
measured before the split would be reading a lane that no longer exists.

Residency is untouched, so the device copy lives exactly as long as it did and
device capacity does not move. What does move is when the spill copy is
written, which is why the split is a proposal rather than a decision: the plan
is simulated again, and it is kept only if the makespan improved. A plan that
did not improve is put back as it was, which the simulation memo prices
without work. So is a plan a configured pool then refuses to place, because a
faster schedule that does not fit is not an answer, and it must not cost the
candidate the answer it had.

Evictions nothing waited on are left alone. Moving such a copy spends lane
time and holds spill capacity longer to buy nothing, and the plan carries an
extra action for it. Two more are never split whatever waited: one whose
object keeps no spill copy, because releasing such an object frees the spill
copy as well, so the split would throw away exactly what the write-back wrote;
and one whose object no task wrote before it, because then the spill copy was
already current and the emitter would have released rather than evicted.

`split_write_backs` selects this, off by default until the corpora say what it
is worth.

### Digest — naming the schedule

Two candidates that reduce to the same plan get the same name. The digest is
how the search recognises a plan it has already measured, and it is what the
shared record carries so a caller can tell whether two searches agree.

### Place — measuring whether the layout fits

Occupancy and extent are different questions. The reducer answers occupancy:
how many bytes are live at an instant. Placement answers extent: how many
bytes the address assignment spans once every lease has a fixed offset. The
extent is the constraint the machine actually imposes, and it is the more
expensive of the two to answer, so the search answers it as rarely as it can.

The shared best-placed record is what makes that affordable. A plan no better
than one already placed cannot become the answer, so it is never measured;
`admits()` is a single atomic read, and the measurement behind it happens
only for plans that could still win. Every plan that could still win *is*
measured, though — skipping on any other ground can leave a candidate that
never placed anything at all, and a candidate with no placed plan has no
answer to give.

A plan whose layout fits is offered to the shared record and kept as this
candidate's answer if it beats what the candidate already placed. A plan whose
layout overruns the pool gives back what it overran, bounded by
`capacity_refinement_bytes`, expresses that smaller capacity to the reducer as
uniform pressure, and plans again from the base residency. Capacity is a
property of the plan, so it travels with the plan and never changes what the
simulator or the caller's budget is. [Capacity
refinement](physical-admission.md#capacity-refinement) is where that trade is
laid out.

### Settle — deciding what to answer with

A plan that never waited for memory is finished, and so is a plan whose
candidate has run out of repairs. Either way the candidate answers. Deciding
that is choosing an answer, so it is reported as `select_ns` — the same
section the problem level uses for adopting its winner.

With a pool to place into, the answer is the best plan whose layout fit — not
the fastest plan simulated. A plan that cannot be placed cannot run, so
offering it as an answer only pushes the rejection to a layer that would have
to walk capacity down to escape it. A candidate that placed nothing reports
`unplaceable`. Without a pool there is nothing to place into, and the
candidate answers with its fastest plan, which is what a caller that supplied
no topology can be told.

### Select — adopting the winner and materialising it

Whatever the shared record holds at the end is the plan the search selected:
selection reads the record rather than ranking the candidates a second time,
because the record already owns a copy of the plan it names. A problem's
winner is the plan to beat it was handed unless a candidate did strictly
better; a tie keeps the plan in hand, so an answer changes only for a reason.
The planner decodes that one indexed schedule, evaluates its physical
admission once more, and materialises the full `SimulationResult` — at the
caller's full capacity, which is the machine the plan will actually run on. A
plan built against a reduced capacity was *chosen* on how it behaves there,
but the reported timeline and the certificate measure the real machine.

The planning store keeps the same promise across runs. The plan in hand is
provenance rather than part of a request's identity, so a request reads back
the plan its search chose whatever it was handed; a stored plan that the plan
in hand claims to beat is searched again with it and replaced only when the
new answer is faster under this request.

### Teardown

Releasing everything the evaluation held. Small, and named so that the time
it takes is attributed rather than left in the residual.

## Trajectories

With `record_reduction_steps` set, each candidate records the cycle rather
than only its outcome: one `ReductionStep` per plan it held, carrying that
plan's makespan, the bytes its layout needed, the capacity it was built
against, the objects the reducer cut to reach it, the repair count, and what
became of it — simulated, measured, placed, refined, best so far, or the
answer. The steps in order are the search itself, which is what a question
like "why is this plan slower than the one at a larger budget" is actually
asking about.

Recording is off by default: it costs an allocation per candidate that grows
with the search, worth paying when attributing planner time or explaining a
plan and not otherwise.

## Pseudocode

The first two lines are the expansion; everything after them is the search
over what it produced.

```text
PressureFit(program, initial, final, machine, options, admission):
    require the planner and simulator ABIs
    resolved = ordered_legal_task_selections(program, options.resolution_options)
    require some selection's anchor/output floor to fit

    best_placed = shared record, empty

    for each resolved program:                      # all in one call, in order
        prepare:  problem = compile_indexed_problem(resolved, machine, admission)
                  seed    = required_anchor_hulls(problem)
                  if options.initial_placement == GREEDY:
                      seed = preplace_fitting_spill_objects(seed)
        setup:    facts, workspace = schedule_facts(problem), allocate()
        incumbent: if the caller handed in a plan for this resolved program:
                      result = simulate(plan, admit(plan))
                      if place_lifetimes(result) fits the pool:
                          best_placed.offer(name, plan); winner = plan

        for strategy in options.residency_strategies:
            reduce:  base = reduce_until_analytic_pressure_fits(seed, strategy)

            for fetch_rule in options.fetch_rules:
                for coalesced in enabled_coalescing_modes:
                    residency = base
                    capacity  = machine.object_capacity
                    placed    = none

                    loop:                            # the candidate cycle
                        emit:      schedule = emit_actions(
                                       residency, fetch_rule, coalesced,
                                       recorded_constraints)
                                   if a constraint cannot hold: stop

                        simulate:  result = simulate(schedule, admit(schedule))

                        repair:    if admission refused the schedule:
                                       advance or delay the fetch, else add the
                                       deficit at the failing boundary and reduce
                                       continue, or stop when nothing is left

                        if result did not simulate:
                            repair:  delay the fetch, else add boundary pressure
                                     and reduce; continue, or stop
                            continue

                        digest:    name = schedule_digest(schedule)

                        place:     if best_placed.admits(result.makespan)
                                          and name not already measured:
                                       extent = place_lifetimes(result)
                                       if extent fits the pool:
                                           best_placed.offer(name, schedule)
                                           placed = better_of(placed, result)
                                       else:
                                           capacity -= min(overrun, refinement)
                                           residency = reduce(seed, strategy,
                                                              pressure=given_back)
                                           continue

                        settle:    if result never waited for memory
                                          or no repairs remain:
                                       answer with placed, or unplaceable
                                   otherwise keep the plan and repair again

    select:   winner = best_placed.read()
              return materialize(winner, at machine.object_capacity)
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
| `relaxed-stall` (not in the default portfolio) | Minimize estimated exposed stall first | No | Currently maps to the same reduction controls as `tight-stall`, so it produces no distinct pressure behavior unless another implementation control is added. It remains a separate candidate identity in diagnostics. |

Headroom accounting charges a fetched residency span one boundary earlier
than its logical entry. It is conservative boundary accounting, not a transfer
start prediction. The transfer lane and exact destination lifetime remain the
simulator and admission model's responsibility.

### Fetch-trigger rules

| Rule | Mechanical behavior | Expected pressure/latency tradeoff |
|---|---|---|
| `packed-fifo` | Work backward from consumer deadlines while packing each device's single fetch lane; earlier fetches account for residual occupancy left by later packed work. | Seeks lane utilization and overlap across the complete reload set. It may enqueue a destination earlier than capacity permits, so admission can delay it or force another residency cut. |
| `packed-fit` | Start with packed FIFO triggers, then move implicated triggers later until their early destination occupancy fits analytic capacity where possible. | Reduces trigger-time capacity pressure relative to unconstrained packing, at the cost of less transfer lead time and potentially more consumer stall. |
| `interval-entry` (not in the default portfolio) | Extend each later residency span toward earlier boundaries while exact analytic capacity fits, then place fetches with packed FIFO. | Uses otherwise idle object capacity to create more transfer lead time. It can hide latency, but increases how long fetched destinations occupy execution memory. |
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
- exhaustive selection of large program-alternative products;
- feasibility after runtime behavior violates the admitted allocation
  contract.

`PlanInfeasibleError` means a necessary-condition preflight failed, or
every candidate across every resolved program was rejected without a
remaining repair path. It does not prove that no schedule outside the
resolved programs searched exists. `PlanSearchExhaustedError` means at
least one repairable path reached its configured repair ceiling, so even
infeasibility across those resolved programs was not established.

## Implementation map

| Layer | Responsibility |
|---|---|
| `shadowspill.planner.search.algorithms.pressurefit` | The search object: input validation, expanding the program into resolved programs and ordering them, and the admission facts stamped onto the answer. It hands every resolution to one call and owns no threads. |
| `shadowspill.planner.search.algorithms.pressurefit.search` | Projecting each resolution into the planner ABI, the preflight that drops the ones that cannot fit, and merging the results into one answer. |
| `csrc/src/planner/residency.c` | Indexed anchor geometry, pressure accounting, legal cuts, scoring, and reduction. |
| `csrc/src/planner/schedule.c` | Gap transitions, fetch-window placement, action emission, and trigger constraints. |
| `csrc/src/planner/candidates.c` | The candidate cycle and its stages, the worker pool and the (resolved program, candidate) tasks it hands out, the memo tables, selection, and section timing. |
| `csrc/src/planner/admission/` | Physical allocation and causal-reuse admission. |
| `csrc/src/planner/best_placed.c` | The shared record of the best plan any search has placed. |
| `shadowspill.simulator` / `csrc/src/simulator` | Independent schedule replay and makespan authority. |

The production path requires the planner and simulator. Readable
Python implementations live only under `reference/python/pressurefit` and
`reference/python/simulator`; they are differential-test oracles and do not
silently replace a missing or ABI-incompatible library.

Previous: [Writing a search algorithm](search-algorithm.md). Next:
[Physical admission and offset handling](physical-admission.md).
