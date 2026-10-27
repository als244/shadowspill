# Tests

The test tree mirrors the code boundary being exercised:

```text
tests/
├── shadowspill/    Python unit/property tests for src/shadowspill
├── csrc/           C and device-backend canaries for csrc, one directory per library
├── integration/    fresh-process framework/backend integration canaries
├── tools/          reusable source-tool tests
├── workloads/      workload/model-definition tests
├── benchmarking/   corpus/frontier harness tests
├── repository/     packaging, naming, and boundary checks
└── fixtures/       immutable test-only golden inputs
```

Inside `shadowspill/`, the mirror goes all the way down: a test for
`src/shadowspill/profiling/` lives in
`tests/shadowspill/profiling/`. A test that exercises two packages
belongs to the one it is about, not the one it imports for a fixture.

`tests/csrc/` mirrors `csrc/` the same way: `runtime/`, `simulator/`, `backends/`,
`adapter/` and `planner/search/algorithms/pressurefit/` hold the canaries for
those libraries, and the two at its root, the ABI sizes and the build, are about
the build as a whole.

Tests requiring the execution allocator run in fresh workers. Build/profile stay
in ordinary PyTorch processes; each execution worker installs the allocator before
creating device tensors. Shared helpers under `tests/shadowspill/execution/`
exercise this boundary and keep process setup out of individual cases.

Long-running numerical and throughput acceptance belongs in `qualification/`,
not in the ordinary unit suite.
