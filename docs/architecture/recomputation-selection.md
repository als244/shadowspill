# Recomputation selection

Recomputation selection constructs the finite family of complete Program task
selections that PressureFit evaluates. It consumes the occurrence-level
options produced by [graph-pair construction](graph-pair-construction.md), but
does not capture, compile, or profile graphs. It is also separate from
[PressureFit](pressurefit.md): recomputation decides **which executable task
alternative is active**, while PressureFit decides **where the resulting
objects reside and when they move**.

The common PyTorch training case gives each differentiated stage occurrence a
`save` graph pair and a `recompute` graph pair. Structurally equivalent
occurrences share graph construction and profiling, but remain separate
Program choices because their lifetimes and surrounding pressure differ. The
framework-neutral IR is more general: a `RecomputationGroup` may expose any
finite set of mutually exclusive `RecomputationOption` values, and Programs
may contain no groups at all.

For example, two occurrences can share one structural portfolio while still
producing independent Program groups:

```text
structural ABI A
    variants = {save: (F_save, B_save), recompute: (F_recompute, B_recompute)}

occurrence group g0 = {save, recompute}
occurrence group g1 = {save, recompute}

complete selection r0 = {g0: save,      g1: save}
complete selection r1 = {g0: recompute, g1: save}
complete selection r2 = {g0: save,      g1: recompute}
...
```

Selection emits a bounded collection of complete rows like $r_0,r_1,r_2$.
It never returns a partial assignment and never treats one structural cache
entry as one global choice for all of its occurrences.

## Inputs and output

The planner consumes only immutable Program facts:

- ordered `RecomputationGroup` values;
- each option's `option_id`, active task IDs, and retained alias IDs;
- task-profile runtimes and alias sizes;
- the selected task dependency graph and task phases.

It returns a tuple of complete `RecomputationSelection` tuples. Each complete
selection chooses exactly one option for every group and becomes one parent
context in PressureFit diagnostics.

The recomputation selector does not consider execution capacity, spill
capacity, transfer bandwidth, residency, or simulated makespan. PressureFit
evaluates those consequences after the portfolio has been built.

## Cost summaries

For group $g$ and option $o$, the current implementation derives two
deterministic summaries:

\[
\operatorname{retained}(g,o)=
\sum_{a\in\operatorname{retained\_aliases}(g,o)}s_a
\]

and

\[
\operatorname{runtime}(g,o)=
\sum_{t\in\operatorname{active\_tasks}(g,o)}c_t.
\]

These summaries order non-binary fallback choices. They are not standalone
makespan estimates because they omit transfer overlap, residency interaction,
and contention; PressureFit's simulator evaluates those jointly.

## Current portfolio algorithm

### No groups

A Program without recomputation groups yields one empty selection. PressureFit
then operates as an ordinary residency and transfer planner.

### Small products

The planner first applies any required terminal-save choices, then computes
the product of the remaining option counts. If the product is at most 64, it
enumerates every legal combination in deterministic group/option order.

### Large binary save/recompute products

If every flexible group has exactly one `save` and one `recompute` option, the
planner emits five coarse selections. Among flexible groups, the target
fractions recomputed are:

```text
0%, 25%, 50%, 75%, 100%
```

For each fraction, recomputed groups are chosen at centered, evenly spaced
positions in stable group order. If there are $G$ flexible groups and the
target chooses $K$, position $j\in\{0,\ldots,K-1\}$ is

\[
\left\lfloor\frac{(2j+1)G}{2K}\right\rfloor.
\]

This distributes recomputation through the graph rather than taking one
contiguous prefix. It is a deterministic coarse portfolio, not an adaptive
search over arbitrary subsets.

### Large non-binary products

For groups with more than the binary endpoints, the fallback constructs and
deduplicates:

- the first option in every group;
- the last option in every group;
- the minimum-retained-bytes option in every group;
- the minimum-profiled-runtime option in every group;
- five within-group retained-memory quantiles at 0%, 25%, 50%, 75%, and 100%.

Retained bytes break ties with runtime and option order; runtime breaks ties
with retained bytes and option order. Required choices are reapplied before
deduplication.

## Terminal forward groups

Every recomputation group whose forward tasks are sinks of the selected
forward dependency graph is required to expose exactly one option named
`save`. That option is pinned in every portfolio selection. Terminal forward
groups are therefore not treated as recomputation degrees of freedom by the
current portfolio.

The rule is graph-derived: it uses task phase and dependency edges, not model
family, module name, stage number, or operator identity.

## Pseudocode

```text
BuildRecomputationPortfolio(program):
    groups = program.recomputation_groups
    if groups is empty:
        return [empty selection]

    required = terminal_forward_sink_save_options(program)
    legal_indices = option indices per group, restricted by required

    if product(len(indices) for indices in legal_indices) <= 64:
        return exhaustive_cartesian_product(legal_indices)

    if every group is exactly {save, recompute}:
        portfolio = []
        for fraction in [0%, 25%, 50%, 75%, 100%]:
            flexible = groups not fixed by required
            chosen = centered_evenly_spaced_subset(flexible, fraction)
            portfolio.append(save_or_recompute_each_group(chosen, required))
        return portfolio

    costs = retained_bytes_and_profiled_runtime(program, groups)
    raw = [
        first_option_per_group,
        last_option_per_group,
        minimum_retained_per_group,
        minimum_runtime_per_group,
        within_group_memory_quantile(0%),
        within_group_memory_quantile(25%),
        within_group_memory_quantile(50%),
        within_group_memory_quantile(75%),
        within_group_memory_quantile(100%),
    ]
    return stable_deduplicate(apply_required_choices(raw))
```

## Scope and limitations

The current algorithm is intentionally bounded and fast. For large binary
products it does not search mixed subsets beyond the five evenly distributed
fractions, and it does not use PressureFit feedback to refine a selection.
Consequently, the best schedule in the emitted portfolio may be worse than a
legal selection that was not emitted.

That limitation belongs here, not inside PressureFit. A richer recomputation
planner can generate a different finite portfolio without changing the
PressureFit Program, residency, action, simulation, or runtime contracts.

Graph-pair construction and profiling are described in the dedicated
[graph-pair construction](graph-pair-construction.md) page. The IR
representation is described in [Intermediate representation](ir.md).

Previous: [Graph-pair construction](graph-pair-construction.md). Next:
[PressureFit](pressurefit.md).
