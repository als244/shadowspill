# Tests

The test tree mirrors the code boundary being exercised:

```text
tests/
├── shadowspill/    Python unit/property tests for src/shadowspill
├── csrc/           C and device-backend canaries for csrc
├── integration/    fresh-process framework/backend integration canaries
├── tools/          reusable source-tool tests
├── workloads/      workload/model-definition tests
├── benchmarking/   corpus/frontier harness tests
├── repository/     packaging, naming, and boundary checks
└── fixtures/       immutable test-only golden inputs
```

Inside `shadowspill/`, the mirror goes all the way down: a test for
`src/shadowspill/pytorch/profiling/` lives in
`tests/shadowspill/pytorch/profiling/`. A test that exercises two packages
belongs to the one it is about, not the one it imports for a fixture.

`tests/shadowspill/pytorch/api/` is the exception, and deliberately so.
Installing ShadowSpill's allocator requires a process where the framework has
not yet initialized its device runtime; PyTorch refuses to swap an allocator
that has already been used. So exactly one of those tests can run per pytest
process. They
sort first, which gives one of them that slot, and each is also registered as
its own CTest so all of them actually run. Do not add a package under
`pytorch/` that sorts before `api`.

Long-running numerical and throughput acceptance belongs in `qualification/`,
not in the ordinary unit suite.
