# Planner qualification tools

The planner qualification pipeline has two deliberately separate collection
stages:

1. `corpus_collection/` performs the expensive model construction,
   Export/AOT/Inductor compilation, task profiling, and Program lowering. It
   emits reusable, self-contained pre-PressureFit `StepProgram` artifacts.
2. `frontier_collection/` repeatedly invokes PressureFit, simulation, and
   physical admission over that frozen input corpus. It emits complete
   `AnnotatedProgramPlan` artifacts plus compact comparison tables.

The thin entrypoints are `collect_corpus.py` and `collect_frontier.py`; strict
versioned configurations live in `configs/`. Shared immutable Program and plan
artifact storage is implemented by `corpus.py`. Diagnostic utilities such as
`compare_legacy.py`, `profile_pressurefit_context.py`, and
`replay_pressurefit.py` inspect those same contracts but are not collection
controllers.

This separation makes planner benchmarks repeatable: new recomputation or
PressureFit code reuses exactly the same compiled/profiled Programs and changes
only the frontier baseline identity.

Annotated plans preserve both a timing-independent semantic plan digest and a
full artifact SHA-256. The latter covers all measured PressureFit, admission,
and orchestration wall times and the complete planner/simulator diagnostics.
