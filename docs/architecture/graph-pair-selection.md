# Graph-pair selection

Graph-pair selection constructs the finite family of complete Program task
selections that PressureFit evaluates. It consumes the occurrence-level
options produced by [graph-pair construction](graph-pair-construction.md), but
does not capture, compile, or profile graphs. It is also separate from
[PressureFit](pressurefit.md): selection decides **which executable task
alternative is active**, while PressureFit decides **where the resulting
objects reside and when they move**.

The common PyTorch training case gives each differentiated stage occurrence a
`save` graph pair and a `recompute` graph pair. Structurally equivalent
occurrences share graph construction and profiling, but remain separate
Program choices because their lifetimes and surrounding pressure differ. The
framework-neutral IR is more general: a `TaskAlternativeGroup` may expose any
finite set of mutually exclusive `TaskAlternativeOption` values, and Programs
may contain no groups at all.

For example, two occurrences can share one set of structural graph pairs while still
producing independent Program groups:

```text
structural contract A
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

## Vocabulary

Five terms name five different things, and they are told apart by what each
one fixes:

| Term | What it is | Who fixes it |
|---|---|---|
| Graph-pair group | One occurrence-level set of mutually exclusive alternatives, `TaskAlternativeGroup`. | [Graph-pair construction](graph-pair-construction.md), one per differentiated stage occurrence |
| Graph-pair option | One alternative in a group, `TaskAlternativeOption`: the tasks it activates and the aliases it retains. | Graph-pair construction |
| Graph-pair choice | One option fixed for one group, `TaskAlternativeChoice`. | Selection, one per group |
| Graph-pair selection | One option fixed for **every** group: one complete row. | Selection, as the bounded family below |
| Graph-pair problem | The question one selection poses to PressureFit, and the diagnostics record its answer produces. | [PressureFit](pressurefit.md), one per selection |

The two that are easiest to confuse differ only in how much they fix:

```text
choice     (g0, save)                          one option, one group
selection  {g0: save, g1: recompute, g2: save} one option, every group
```

Two neighbouring terms are deliberately outside the family. A **candidate
policy** — a residency strategy, fetch rule, and coalescing mode — is
evaluated *within* one problem, so a plan is chosen by a selection and a
policy together, and "selection" alone never means the policy.
**Recomputation** keeps its own name wherever it is the subject: a graph pair
either recomputes or it does not, and `recomputation_overhead_seconds` is the
compute a selection spends above the cheapest option of every group.

### Where the name applies

A graph pair is a forward graph and its backward graph, so this whole family
is training vocabulary. The neutral tree does not use it. `shadowspill.ir`,
`shadowspill.planner`, `shadowspill.simulator` and `shadowspill.runtime` plan
any **resolved program** — a Program with every alternative fixed, leaving one
concrete task set — and an inference Program resolves a forward partition with
no pair in sight. So the IR names the general case: a `TaskAlternativeGroup`
owns mutually exclusive `TaskAlternativeOption` values, and a
`TaskAlternativeChoice` fixes one option for one group.

A graph-pair selection is the training-specific instance of that. The frontend
builds each group's options from the forward/backward pairs of one structural
contract, so a complete selection resolves the program by fixing one pair per
occurrence. That is why [PressureFit](pressurefit.md) and [from a resolved
program to leases](admission-leases.md) are written in terms of resolved
programs while this page is written in terms of graph pairs: the same object,
named from whichever side of the boundary is speaking.

The serialized keys spell three of these differently — `recomputation_groups`
in the Program JSON, `recomputation_selection` and `recomputation_problems` in
the plan store. Those keys are identity rather than prose: `Program.digest` is
computed over the Program's own JSON, and every plan-store key derives from
it, so a key moves only with a schema version and a recollected corpus.

## Inputs and output

The planner consumes only immutable Program facts:

- ordered `TaskAlternativeGroup` values;
- each option's `option_id`, active task IDs, and retained alias IDs;
- task-profile runtimes and alias sizes;
- the selected task dependency graph and task phases.

It returns a tuple of complete `TaskAlternativeChoice` tuples. Each complete
selection chooses exactly one option for every group and becomes one parent
problem in PressureFit diagnostics.

The graph-pair selector does not consider execution capacity, spill
capacity, transfer bandwidth, residency, or simulated makespan. PressureFit
evaluates those consequences after the resolutions have been built.

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

## The current selection policy

### No groups

A Program without graph-pair groups yields one empty selection. PressureFit
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
contiguous prefix. It is a deterministic coarse policy, not an adaptive
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

Every graph-pair group whose forward tasks are sinks of the selected
forward dependency graph is required to expose exactly one option named
`save`. That option is pinned in every resolution. Terminal forward
groups are therefore not treated as recomputation degrees of freedom by the
current policy. The rule names the `forward` phase deliberately: a Program
that declares no forward phase pins nothing and keeps every alternative open.
See [phases and sinks](ir.md#phases-and-sinks) for what sink means and why
generalising the rule would be worse than naming the phase.

The rule is graph-derived: it uses task phase and dependency edges, not model
family, module name, stage number, or operator identity.

## Pseudocode

```text
Resolutions(program):
    groups = program.task_alternative_groups
    if groups is empty:
        return [empty selection]

    required = terminal_forward_sink_save_options(program)
    legal_indices = option indices per group, restricted by required

    if product(len(indices) for indices in legal_indices) <= 64:
        return exhaustive_cartesian_product(legal_indices)

    if every group is exactly {save, recompute}:
        resolutions = []
        for fraction in [0%, 25%, 50%, 75%, 100%]:
            flexible = groups not fixed by required
            chosen = centered_evenly_spaced_subset(flexible, fraction)
            resolutions.append(save_or_recompute_each_group(chosen, required))
        return resolutions

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
Consequently, the best schedule among the emitted resolutions may be worse than a
legal selection that was not emitted.

That limitation belongs here, not inside PressureFit. A richer recomputation
planner can generate a different finite family of resolutions without changing the
PressureFit Program, residency, action, simulation, or runtime contracts.

Graph-pair construction and profiling are described in the dedicated
[graph-pair construction](graph-pair-construction.md) page. The IR
representation is described in [Intermediate representation](ir.md).

Previous: [Graph-pair construction](graph-pair-construction.md). Next:
[PressureFit](pressurefit.md).
