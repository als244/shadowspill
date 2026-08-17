# Tests

The test tree mirrors the code boundary being exercised:

```text
tests/
├── shadowspill/    Python unit/property tests for src/shadowspill
├── csrc/           compiled C/CUDA canaries for csrc
├── integration/    fresh-process framework/backend integration canaries
├── tools/          reusable source-tool tests
├── workloads/      workload/model-definition tests
├── benchmarking/   corpus/frontier harness tests
├── repository/     packaging, naming, and boundary checks
└── fixtures/       immutable test-only golden inputs
```

Long-running numerical and throughput acceptance belongs in `qualification/`,
not in the ordinary unit suite.
