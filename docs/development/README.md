# Development guide

- [Repository structure and validation](repository.md) — the Python and C
  trees, where a test for a given module goes, setup, and the lint, type and
  build commands a change has to pass.
- [Naming conventions](naming.md) — the vocabulary every surface shares, the
  words to avoid and why, and what stays generic outside a backend.

The [Python](../python/README.md) and [C](../c/README.md) references document
the surfaces a change must keep working.

Product code belongs in `src/shadowspill/` or `csrc/`. Reusable source-tree
tooling belongs in `src/tools/`. Workload definitions, planning benchmarks,
and release qualification consume product APIs and do not contain alternate
implementations.

Public documentation is normative and describes only shipped behavior.
Root-cause narratives, engineering plans, and logs belong under the ignored
internal documentation tree, `docs/internal/`.
