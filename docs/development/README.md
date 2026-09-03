# Development guide

- [Repository structure and validation](repository.md)
- [Naming conventions](naming.md)
- [Python documentation](../python/README.md)
- [C documentation](../c/README.md)

Product code belongs in `src/shadowspill/` or `csrc/`. Reusable source-tree
tooling belongs in `src/tools/`. Workload definitions, planning benchmarks,
and release qualification consume product APIs and do not contain alternate
implementations.

Public documentation is normative and describes only shipped behavior.
Root-cause narratives, engineering plans, and logs belong under the ignored
internal documentation tree, `docs/internal/`.
