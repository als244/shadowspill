# Development guide

What a change to this repository has to fit: where its code and tests belong,
what it must pass, and the words every surface uses.

- [Repository structure and validation](repository.md) — the Python and C
  trees, where a test for a given module goes, setup, and the lint, type and
  build commands a change has to pass.
- [Naming conventions](naming.md) — the vocabulary every surface shares, the
  words to avoid and why, and what stays generic outside a backend.

The [Python](../python/README.md) and [C](../c/README.md) references document
the surfaces a change must keep working.

Product code belongs in `packages/engine/src/shadowspill/` or `packages/engine/csrc/`. Reusable source-tree
tooling belongs in `tools/`. Workload definitions, planning benchmarks,
and release qualification consume product APIs and do not contain alternate
implementations.

Public documentation is normative and describes only shipped behavior.
Root-cause narratives, engineering plans, and logs belong under the ignored
internal documentation tree, `docs/internal/`.

## Documentation ownership

Keep one authoritative reference per subject and link it from examples and
guides. Entry pages should help readers choose a task, not enumerate every
implementation detail. Keep the supported training guide in
`docs/python/client.md`, the custom-model and phase guides under
`docs/python/`, and specialized guides under `docs/python/advanced/`.
Put signatures and constraints under `docs/python/api/`, design invariants under
`docs/architecture/`, and library-specific adapters under `docs/integrations/`.
Keep the native API in `docs/c/` and complete examples under `docs/examples/`.

Use current behavior and concrete names. Explain what a control changes and
which phase consumes it. Preserve shape, ownership and compatibility limits;
link their reference instead of repeating long qualifications in every guide.
Keep historical measurements and investigations under `docs/internal/`.

Run `python -m pytest -q tests/repository/test_documentation.py` after editing.
The checks cover public Markdown links and anchors, source signatures, Python
example syntax and keywords, API export coverage, and documentation reachability.
They do not replace execution tests for examples that require a device.
