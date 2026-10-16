# PressureFit

PressureFit searches for a memory schedule for a fixed task order. It starts
with objects resident between their required uses, cuts residency where memory
is over budget, turns the resulting gaps into transfers, and measures the
schedule. Admission, simulation, and placement can send it back to change a
transfer or make more room. The answer is the fastest valid plan it found.

This page describes the algorithm. The surrounding planner contract is in
[plan search](search.md), the option types are in the
[Python API](../python/api/neutral.md), and the
[C API](../c/pressurefit.md) describes the indexed interface.

## Built-in candidate policies

A **candidate** is one resolved program plus a residency strategy and fetch
rule:

| Choice | Where it acts | What it decides |
|---|---|---|
| Residency strategy | Initial reduction and every reduction requested by a repair | How to account for resident bytes, and which legal cut to make at an over-budget boundary. |
| Fetch rule | Schedule emission, after reduction has chosen the resident spans | At which legal task boundary to enqueue each required fetch. |

The four distinct residency strategies are **names for combinations of two
independent controls**. They are not four different reduction algorithms.
The default search uses the two stall strategies and four fetch rules below:
**eight policies per resolution**. All use the split eviction/fetch exposure
score and the fixed `E, writeback, -slack, -size` ranking, with stable identity
tie breaks. Scoring is part of the algorithm, not another policy option.

| Memory accounting / Cut ranking | Estimated exposed stall first | Write-back requirement first |
|---|---|---|
| Charge the span without an extra boundary before its start | `tight-stall` | `tight-transfer` |
| Also charge one boundary before a fetched span starts | `headroom-stall` | `headroom-transfer` |

The first control changes which boundaries appear over budget. The second
changes which cut wins at the chosen boundary. Both feed the same greedy
reducer. The suffix `transfer` means “omit the stall estimate from the ranking”;
it does **not** mean “minimize total transferred bytes.” The exact accounting
and ranking are defined below.

The fetch rules use the same spans but choose different enqueue boundaries:

| Fetch rule | Trigger selection | What it accounts for |
|---|---|---|
| `demand` | Last legal boundary before the consumer | Legal ordering only. |
| `latest-safe` | Latest legal boundary allowing this fetch to finish by its ideal deadline | This fetch's estimated duration, considered independently. |
| `packed-fifo` | Pack fetch durations backward from their deadlines, then map starts to legal boundaries | Other fetches competing for the same FIFO lane. |
| `packed-fit` | Start with `packed-fifo`, then delay selected fetches at over-budget boundaries | Packed timing plus analytic memory occupied by early fetch destinations. |

For example, `headroom-stall/packed-fit` means: reduce with the extra
boundary charge and stall-first ranking; emit with packed fetches followed by
the capacity pass. A repair can change
the spans or constrain the triggers, so it can invoke these policies again.

## Inputs

PressureFit consumes a [program](program.md), declared initial and final
residency, machine capacities and bandwidths, and `SearchOptions`. Task facts
include inputs, outputs, mutations, dependencies, profiled runtimes, and
workspace. Alias facts include sizes, access/write histories, and whether a
spill copy may be retained. Names and framework-specific metadata do not
affect the search.

Optional `AdmissionFacts` describe physical allocation and storage ownership,
pool capacity, and alignment. The physical pool capacity and the logical
object capacity offered to the reducer are separate quantities. A caller can
also supply an incumbent plan to beat.

## Current algorithm

A resolved program fixes the program's [task alternatives](program.md#task-alternatives).
PressureFit expands the configured resolutions, orders them by descending
share of groups choosing recomputation (ties retain expansion order), and
evaluates each policy combination for each one. Different task orderings or microbatch geometries
are different input programs, chosen by the surrounding caller.

## Pseudocode

```text
PressureFit(program, machine, options, optional_admission, optional_incumbent):
    problems = prepare configured resolutions of program
    discard resolutions whose required memory floor cannot fit
    measure the incumbent, if present, at this capacity

    for each (problem, strategy, fetch_rule):
        # This whole tuple is one worker job, including its base reduction.
        policy = (strategy, fetch_rule)
        candidate_answer = EvaluateCandidate(problem, policy)
        retain the best valid answer for this problem

    select the best answer across problems and the eligible incumbent
    materialize its schedule and full simulation at the caller's capacity

EvaluateCandidate(problem, policy):
    extra_pressure = 0
    trigger_constraints = empty
    best_answer = none
    residency = Reduce(problem.seed, policy.strategy, extra_pressure)

    while residency is feasible:
        schedule = Emit(residency, policy.fetch_rule)
        apply recorded trigger_constraints; stop if they cannot hold
        result = Simulate(schedule, optional physical admission)

        if admission or simulation failed:
            try a legal fetch move; otherwise add measured boundary pressure
            if no repair is possible or its attempt limit is reached: stop
            if pressure changed: residency = Reduce(seed, strategy, extra_pressure)
            continue

        if physical placement is required:
            if this schedule can improve the placement bound:
                measure the fixed-offset layout
                if it fits: retain it if better than best_answer
                else if a useful layout-driven fetch delay is available:
                    record the delay; continue
                else:
                    add uniform pressure for the layout overrun
                    clear trigger_constraints for the new capacity round
                    residency = Reduce(seed, strategy, extra_pressure)
                    continue
        else:
            retain this simulated plan if better than best_answer

        if result did not wait for memory, or no further repair is possible:
            stop
        repair the measured memory shortfall, moving a fetch or adding pressure
        if pressure changed: residency = Reduce(seed, strategy, extra_pressure)

    return best_answer, or a failure if no acceptable plan was found
```

Each reduction starts again from the seed with the current pressure inputs.
Fetch-only repairs reuse the current residency. A candidate keeps its best
acceptable schedule, so its answer may come from an earlier iteration rather
than the last one it explored. The detailed repair and placement rules appear
after the policy definitions.

## Mathematical formulation

### Tasks, boundaries, and the seed

For one resolved program, tasks are numbered $0,\ldots,n-1$. Boundary $-1$
is before task 0; boundary $i$ is after task $i$. Task $i$ consumes inputs at
boundary $i-1$ and publishes outputs at boundary $i$.

An **alias bundle** is a set of logical objects sharing storage. Memory is
counted once per bundle. Let $s_a$ be alias $a$'s size and $H_a$ its required
residency anchors: input/mutation boundaries, output/mutation boundaries, and
required initial or final device residency. The seed is one inclusive span
per alias with any anchors:

$$
R_a^0=[\min H_a,\max H_a].
$$

A residency $R_a$ is an ordered list of inclusive spans. Cuts preserve every
anchor. A normal cut removes a whole available anchor-free run within a span.
A cut can also split a span immediately after an anchor whose last associated
access has completed. This second operation changes the span segmentation
without removing any covered boundary; it creates a legal departure/re-entry
point. The distinction matters when replaying cuts.

The implementation and trace arrays use **positions** $p=b+1$: position 0 is
before task 0. Cut endpoints in the trace retain the signed boundary notation.
See [trace positions and semantic labels](../python/pressurefit-trace.md#semantic-labels-and-positions).

### Prepare — deriving the residency problem

Preparation derives anchor lists, fresh-output reservations, access/write
lists, boundary capacities, and estimated transfer durations. It checks that
required inputs, outputs, and workspace can coexist. A failing resolution is
reported separately; other resolutions can still be searched.

Objects smaller than `minimum_object_bytes_evict_eligible` are assigned to a
[resident slice](fixed-placement.md#the-resident-slice). Its capacity is
reserved before reduction. These objects have fixed fetch triggers, are not
cut, and are excluded from the reducer's variable pressure sum.

Initial residency remains the caller's actual state. An input initially in
spill memory requires an ordinary fetch. A zero-duration control task in
generated training programs supplies an action boundary before the first
computation; there is no free promotion of spill inputs to device memory.

The heuristic clock uses profiled task runtimes $c_i$:

$$
T_{-1}=0,\qquad T_i=\sum_{j=0}^{i}c_j.
$$

$T_i$ is the ideal end of task $i$, before transfer or capacity stalls. The
estimated duration of a fetch of alias $a$ is

$$
f_a=\left\lceil\frac{10^9s_a}{\max(B_{\mathrm{fetch,solo}},B_{\mathrm{fetch,concurrent}})}\right\rceil.
$$

The eviction estimate $v_a$ uses the corresponding eviction bandwidths.
Durations are in nanoseconds. Route latency is excluded from these estimates.
These are local estimates; simulation measures the resulting schedule with
its actual lane contention and task readiness.

## Reduce — choosing what stays resident

The equations describe pressure on a given device. The reducer maintains
separate pressure and capacity for each device and cuts aliases on the
selected device. Device indices are omitted below for readability.

### Control 1: how residency counts toward memory pressure

For a span $[s,e]$, the `tight` control begins charging at $s$. The `headroom`
control begins at $s-1$ if $s>-1$ and the alias is not produced at $s$:

$$
\operatorname{chargedStart}(a,[s,e],h)=
\begin{cases}
s-1,& h=1,\ s>-1,\ \text{and the span needs a fetch},\\
s,& \text{otherwise}.
\end{cases}
$$

Here $h=1$ selects `headroom`, and $h=0$ selects `tight`. For example, a
fetched span beginning at boundary 4 starts contributing bytes at boundary 4
under `tight`, and at boundary 3 under `headroom`.

**Headroom is one extra boundary of memory accounting.** It does not choose a
fetch trigger, estimate how many boundaries a transfer takes, or guarantee
room for the entire early fetch. The fetch rule may enqueue much earlier;
admission and simulation still check the resulting reservations.

Both controls use the same end adjustment: do not charge the span's final
boundary $e$ when $e\geq0$, final device residency is not required, and no
access associated with that span remains after $e$. The object can depart
before the next task is admitted. Otherwise charge through $e$.

Fresh outputs are reserved at the boundary **before** their producing task,
even though their residency anchor is after that task. Let $U_a(R,h)$ be the
union of alias $a$'s charged spans and these output-reservation boundaries.
Overlaps count once. If $\mathcal A_c$ contains this device's cuttable aliases, analytic
pressure is

$$
M_b(R,h)=\sum_{a\in\mathcal A_c}s_a\,\mathbf 1[b\in U_a(R,h)]+X_b.
$$

$X_b$ is additional pressure accumulated by repairs, initially zero. If $C$
is the object capacity left after reserving the resident slice, the boundary
capacity is

$$
C_b=
\begin{cases}
C-w_{b+1},& b+1<n,\\
C,& b=n-1,
\end{cases}
$$

where $w_i$ is task $i$'s workspace, or zero for a task on another device.
This charges workspace only where that
task needs it. The reducer seeks $M_b\leq C_b$ at every boundary. Physical
fragmentation, transfer lifetimes, and exact lane timing require later checks.

### Choosing the boundary and applying cuts

```text
Reduce(seed, strategy, extra_pressure):
    R = copy(seed)
    compute pressure using strategy's tight/headroom control
    while some boundary is over capacity:
        b = boundary with greatest (pressure[b] - capacity[b])
            breaking ties by earlier boundary
        legal = active cuts that relieve b while preserving all anchors
        if legal is empty: return analytically infeasible
        cut = lexicographic minimum under strategy's cut ranking at b
        apply cut to R; update that alias's pressure contribution
        refresh which cuts for that alias remain available
    return R
```

For multiple devices, boundary selection breaks equal excesses by boundary,
then device priority, then device index. The implementation uses a pressure
heap and a segment tree of cuts covering each boundary. The slack
ranking examines every active entry at the chosen boundary: both exposure
and slack depend on that boundary, so a zero exposure score alone does not
justify stopping. This selects the exact minimum without a dense
alias-by-boundary score cache.

### Control 2: which legal cut wins

Write a cut as $(a,\ell,u)$: it removes boundaries $\ell$ through $u$ of alias
$a$. A pure span break has $\ell=u+1$ and length zero. Define:

| Quantity | Definition |
|---|---|
| $W$ | 0 for an initial-prefix cut (already in spill). Otherwise 1 if no spill copy may be retained, or a declared write occurred by departure boundary $\ell-1$; otherwise 0. |
| $D$ | Ideal **completion** of the last task accessing the alias on the cut's left; 0 when there is no prior use. Input and produced-output anchors both use their actual task identity. |
| $N$ | Ideal **start** of the first task accessing the alias on the cut's right; timeline end when only final device residency remains. |
| $B,F$ | Ideal begin and finish of the task immediately after the selected pressure boundary. At the terminal boundary both equal timeline end. |

The eviction and fetch have different overlap windows:

$$
S_{\mathrm{evict}}=\max(0,B-D),\qquad
S_{\mathrm{fetch}}=\max(0,N-F).
$$

$$
E_{\mathrm{evict}}=W\max(0,v_a-S_{\mathrm{evict}}),\qquad
E_{\mathrm{fetch}}=\max(0,f_a-S_{\mathrm{fetch}}).
$$

$$
E=E_{\mathrm{evict}}+E_{\mathrm{fetch}}.
$$

Eviction can overlap compute after the prior use finishes and before memory
is needed for the pressure task. A clean release has zero eviction cost.
Fetch overlap starts after that pressure task finishes. These separate
`max(0, ...)` terms prevent unused time on one side from canceling a real
delay on the other. The same cut can therefore have different scores at
different pressure boundaries.

This is a local estimate without queue contention. It neither simulates all
transfers nor maintains the emitter's refreshed-spill-copy history. In
particular, $W$ remains a conservative write-back flag; the emitter decides
the actual departure action. The score does not add a hard fetch-trigger
constraint: fetch rules can still choose earlier legal triggers, and the
existing simulator/placement/repair loop checks their memory effects.

The ranking also compares **signed slack**:

$$
\sigma=S_{\mathrm{evict}}-Wv_a+S_{\mathrm{fetch}}-f_a.
$$

Slack is maximized. It is not clamped: positive spare time
and negative deficits remain distinguishable. Unlike $E$, slack can let
surplus overlap on one transfer offset a deficit on the other, which is why
it only breaks ties after $E$. Clean releases set $Wv_a=0$ but retain the
eviction-side window in the requested formula. Queue contention is absent
from both metrics.

The keys rank **lexicographically, ascending**. The first differing
component decides; these terms are not added or weighted:

$$
K_{\mathrm{stall}}=(E,\ W,\ -\sigma,\ -s_a,\ a,\ \ell).
$$

$$
K_{\mathrm{transfer}}=(W,\ -\sigma,\ -s_a,\ a,\ \ell).
$$

If the whole tuple ties, the lower internal cut ID wins. Thus `stall` prefers
less estimated exposed time first; `transfer` skips that comparison. Both
then prefer no write-back, more signed slack, larger size, lower alias index,
and earlier cut start, in that order. The size term favors more relief per
cut, not fewer bytes transferred.

The signed slack calculation uses a wide exact integer representation; it
neither overflows nor saturates to fit a signed 64-bit value. This ranking is
fixed for every reduction and candidate. Transfer strategies omit only the
leading exposure term; all other priorities remain the same.

The trace records the winning tuple and up to three next-ranked legal cuts
at the chosen boundary. It also records the three largest boundary excesses
before the cut. These answer different questions: **where pressure was worst**
and **why this cut won there**.

## Emit — turning spans into transfers

A gap between spans creates a departure and a later fetch unless the later
span begins with production. A current, retained spill copy permits release;
otherwise a value needed again requires eviction. The emitter tracks writes
since the spill copy was refreshed. Final residency determines whether the
last span is retained, released, or evicted.

### Legal fetch windows

For reload $r$ of alias $a$, let $[E_r,L_r]$ be its inclusive legal trigger
window and $e_r$ the later span's entry boundary.

- $L_r$ is the boundary immediately before the span's first consumer. For a
  span with no task consumer, it is the terminal task boundary.
- The first fetch has $E_r=0$. After a departure at boundary $p$, emission
  uses $E_r=p+1$.
- Resident-slice objects have $E_r=L_r$ at their previously fixed trigger.

An empty window is infeasible. A first consumer at task 0 has no preceding
action boundary; a spill input therefore needs the control-task arrangement
described above.

Every rule chooses a trigger $k_r\in[E_r,L_r]$. This is an **enqueue boundary**,
not an exact wire-start timestamp. The fetch lane is FIFO, and destination
memory is reserved at the trigger even if the copy must wait behind others.
The heuristic consumer deadline is $T_{L_r}$.

Define the common helper

$$
\operatorname{Latest}(E,L,t)=
\begin{cases}
\max\{i\in[E,L]:T_i\leq t\},& \text{if this set is nonempty},\\
E,& \text{otherwise}.
\end{cases}
$$

The fallback picks the earliest legal boundary even when there is insufficient
lead time. “Safe” in the rule name is an estimate, not a feasibility guarantee.

### `demand`

$$
k_r=L_r.
$$

The fetch is enqueued as late as legality allows. This limits early destination
occupancy, but usually exposes the transfer to its consumer.

### `latest-safe`

Each reload is considered independently:

$$
k_r=\operatorname{Latest}(E_r,L_r,\max(0,T_{L_r}-f_a)).
$$

There is no accounting for another fetch occupying the lane. Two reloads can
each appear to meet a deadline and still queue behind each other.

### `packed-fifo`

Process reloads in descending latest-trigger order, breaking ties by descending
alias index, then ascending original emission ordinal. Pack a separate time
cursor for each device's fetch lane:

```text
packed_start[device] = infinity
for r in sort(reloads, key=(-latest_trigger, -alias, ordinal)):
    finish  = min(T[L_r], packed_start[device(r)])
    desired = max(0, finish - fetch_duration[r])
    k_r     = Latest(E_r, L_r, desired)
    packed_start[device(r)] = max(T[k_r], desired)
```

The last line preserves a desired start lying between task boundaries; it
does not round the packing cursor all the way back to the chosen trigger.
This rule accounts for other planned fetch durations, but it does not check
destination capacity while choosing triggers.

For example, suppose legal boundary times are 0, 5, 10, and 15, and two fetches
each take 8 time units with deadline 15. `latest-safe` chooses time 5 for both.
Packing chooses time 5 for the first processed fetch and time 0 for the next,
creating more lead time at the cost of earlier destination occupancy.

### `packed-fit`

First run `packed-fifo`. Then add early fetch destinations to the analytic
pressure and move selected triggers later. For one device, let

$$
A_{r,b}=\mathbf 1[k_r\leq b<e_r\ \text{and alias }a(r)\text{ is not logically resident at }b].
$$

The capacity pass uses

$$
P_b=M_b(R,h)\big|_{X_b=0}+\sum_r s_{a(r)}A_{r,b}.
$$

It uses the strategy's span charges and fresh-output reservations, without
the reducer's accumulated repair pressure $X_b$.

```text
for b in task boundaries 0 .. n-1, ascending:
    while P[b] > C[b]:
        movable = reloads with A[r,b] = 1 and k_r < L_r
        if movable is empty: break
        r = first by (entry boundary descending, size descending,
                      alias descending, packed-array index ascending)
        new_trigger = min(b + 1, L_r)
        remove r's early destination charges before new_trigger
        k_r = new_trigger
```

This pass does not repack the lane after delaying a fetch. It can leave a
boundary over capacity when no trigger can move, and a delay can expose
consumer stall. Admission, simulation, and repair decide what follows.

### Repair constraints

After emission, recorded repair constraints can move triggers within their
legal bounds. A schedule that cannot honor them stops that trajectory. Thus
the final trigger need not be the fetch rule's initial choice.

## Simulate and repair

Simulation measures the emitted schedule, including task readiness, transfer
lanes, and capacity waits. With admission facts, physical admission runs
inside this step: it derives allocation lifetimes, reservations, retirements,
and causal reuse using the production memory-pool policy. See
[simulation](simulation.md) and [physical admission](physical-admission.md).

A schedule can finish while waiting for device memory. That gives a valid
makespan and a measured shortfall worth trying to remove. An impossible
initial state, spill-pool overflow, or deadlock can instead fail simulation.

For a repairable **admission failure**, try:

1. Advance a fetch to a compatible release boundary.
2. Otherwise delay a fetch toward its consumer.
3. Otherwise add the measured deficit at the failing boundary to $X_b$ and
   reduce again from the seed.

For a repairable **simulation capacity failure or capacity wait**, try a
fetch delay, then a boundary-pressure repair. Repeated failures at the same
task and time can double the pressure request, up to the whole allocation
request. If the escalated request cannot be reduced, it is taken back and the
plain deficit is tried. Fetch moves and pressure repair attempts are bounded
by `max_repair_attempts`; non-capacity contradictions are rejected directly.

These operations explain why a trace contains several reductions and several
emitted schedules. A pressure repair reruns the greedy reducer; a trigger
repair can produce a new schedule without making a new cut. The trajectory
is not guaranteed to improve makespan at every step.

## Place — measuring whether the layout fits

Occupancy and address extent are different: even when live bytes fit, a fixed
offset assignment can extend beyond the physical pool because of overlapping
lifetimes and fragmentation. Placement measures that extent.

The placement gate skips a schedule that cannot beat an already placed plan.
Normally the bound is shared across candidates and resolutions. With
`deterministic=True`, the gate consults only this candidate's own placed
plans. A schedule whose layout fits is eligible to become the answer.

When the layout overruns the pool, PressureFit first tries delaying a fetch
whose destination contributes to the overrun. The move must free at least the
overrun, remain before its consumer, and, after a previous such move, follow
an actual reduction in measured extent. Otherwise it refines capacity.

For overrun $o$ and configured refinement increment $q$, the give-back is

$$
\delta=
\begin{cases}
\min(o,q),& q>0,\\
o,& q=0.
\end{cases}
$$

The candidate reduces its recorded planning capacity by $\delta$, adds
$\delta$ to every boundary's extra pressure, clears the fetch constraints,
and reduces again. This is a
separate capacity-refinement round, counted by `capacity_refinements`. It
does not change the caller's physical budget. See
[capacity refinement](physical-admission.md#capacity-refinement).

## Settle and select

With placement, a candidate answers with its fastest placed schedule; without
placement, with its fastest valid simulated schedule. It can restore an
earlier answer after a later repair fails or makes the plan slower. A
candidate that never obtained an acceptable answer reports failure.

Each resolved problem retains its best candidate. Equal makespans favor the
earlier candidate ordinal; a valid incumbent wins a tie against a newly
generated candidate. Across resolutions, equal makespans favor the earlier
resolution. The shared best-placed record supplies the placement bound; the
per-problem results retain the schedules from which the final answer is
materialized.

The selected schedule is simulated again at the caller's full capacity. Its
reported task/transfer timeline is therefore for the machine that will run
it, even if the search selected it during a reduced-capacity round. Requested
per-resolution answers are materialized the same way.

PressureFit finds the best plan reached by this bounded search. It does not
exhaust all task selections, possible cuts, or trigger assignments.

## Output

`ProgramPlanResult` contains the chosen task selections, memory schedule,
full simulation, search diagnostics, and resident-slice information. It also
preserves the program, residency declarations, machine configuration, search
options, and any admission/placement facts needed to explain the answer.
Requested per-resolution results accompany the overall winner.

If no candidate answers and at least one exhausted its repair budget, the
search raises `PlanSearchExhaustedError`. Otherwise a failed preflight or no
acceptable candidate raises `PlanInfeasibleError`, with available failure
details and diagnostics. A search failure does not prove that every possible
schedule is infeasible.

## Options and execution

| PressureFit option | Default | Role |
|---|---|---|
| `resolution_options` | Shares 0, 0.25, 0.5, 0.75, 1 | Which task-alternative resolutions to search. |
| `residency_strategies` | `headroom-stall`, `tight-stall` | Memory accounting and cut ranking. |
| `fetch_rules` | `packed-fifo`, `packed-fit`, `latest-safe`, `demand` | Fetch-trigger assignment. |
| `max_repair_attempts` | 256 | Bound fetch/pressure repair attempts. |
| `capacity_refinement_bytes` | 256 MiB | Bound the give-back in each layout-capacity round; zero takes the whole overrun. |
| `record_reduction_steps` | `False` | Save schedule-level search steps in diagnostics. |

The default portfolio has $2\times4=8$ plain policies per resolved program.
Five distinct resolutions therefore produce 40 candidate jobs. The two
`transfer` strategies are available as explicit alternatives to the default
stall-first strategies.

### Workers and the unit of work

One worker job evaluates one **(resolved program, policy)** pair, including
its own base reduction. Prepared problem facts are shared; workspaces are
reused by workers. Base reductions are not shared across fetch-rule
candidates, even when their residency strategy matches.

`workers` controls a per-call pool, independently of the number of resolved
programs. In ordinary mode, workers observe the shared placement bound at
different times; this can change the explored trajectories. `deterministic`
removes that dependency from each candidate's placement gate and makes its
outcome reproducible across worker counts. The incumbent can still win the
final comparison.

## Diagnostics and cut traces

Ordinary diagnostics include all evaluated resolutions and candidate outcomes,
repair and placement counters, schedule digests, section timings, and elapsed
start/end times. `record_reduction_steps` adds one record per schedule-level
step: makespan, planning capacity, required layout bytes, repair count, and
what happened to the plan.

`SearchOptions.trace_directory` adds the detailed
[PressureFit cut trace](../python/pressurefit-trace.md). The normal API finds
the winner first and reruns that candidate deterministically for recording.
Explicit resolution/policy pairs can request more candidates without changing
the selected plan or its cache key. Each trace includes:

- the seed and every cut, with enough pressure updates to replay the spans;
- winning and rejected cut rankings, and the top three over-budget boundaries;
- links between reductions, emitted-plan steps, and the kept answer;
- task profile durations and the returned plan's task timings;
- alias members and optional semantic object/task labels for later analysis.

Semantic names such as forward stage, microbatch, parameter path, or activation
origin are annotations. They do not enter policy decisions. Stage summaries
can be derived from task timings; cut order is the planner's decision order,
not the execution timeline.

The disjoint timing sections are `prepare`, `setup`, `reduce`, `emit`,
`simulate`, `repair`, `digest`, `place`, `select`, and `teardown`, plus
`residual_ns`. Initial reduction is charged to `reduce`; reductions requested
by repairs are charged to `repair`. `admit_ns` is nested inside `simulate_ns`.
Summed worker time can exceed elapsed call time. The
[C diagnostics reference](../c/pressurefit.md#diagnostics) gives the exact
counter definitions.

## Implementation map

Paths below are relative to `csrc/src/planner/search/algorithms/pressurefit/`:

| Component | Source |
|---|---|
| Anchors, seed, task clock, transfer estimates, resident slice | `problem/` |
| Strategy-to-control mapping | `candidates/plan.c` |
| Boundary selection and pressure accounting | `residency/heap.c`, `residency/spans.c` |
| Legal cuts and lexicographic ranking | `residency/cuts.c` |
| Fetch windows and departures | `schedule/emit.c` |
| Latest-safe and packed triggers | `schedule/triggers.c` |
| Packed-fit capacity pass | `schedule/clamp.c` |
| Candidate cycle, repairs, capacity refinement, kept answer | `candidates/search.c` |
| Per-candidate jobs and per-problem winner | `candidates/workers.c` |

Python resolution expansion and final materialization live in
`src/shadowspill/planner/search/algorithms/pressurefit/search.py`.
