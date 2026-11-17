# Tests

The test tree mirrors the code boundary being exercised:

```text
packages/client/tests/client_tests/
└── ...             portable configuration, packing, transport and client behavior

packages/engine/tests/engine_tests/
├── build/          capture/compiler checks (other subpackages mirror the engine)
├── native/         C and device-backend canaries
├── integration/    fresh-process framework/backend canaries
├── orchestration/  launch, monitoring and capture tests
└── fixtures/       immutable test-only golden inputs

packages/workloads/tests/workload_tests/
├── models/         architectures and family registration
├── initialization/ checkpoint mapping, loading and dtype policy
├── definitions/    public model/objective registration and preparation settings
├── inputs/         runtime sequence metadata and packed input execution
├── objectives/     losses and fused head math
├── policies/       chunking and retention behavior
├── providers/      operation-library integration and prepared_sonic fixtures
├── service/        public client training, reports and continuation
│   ├── helpers/    subprocess client/server entry points
│   └── oracles/    independent numerical references for those integrations
└── cases/          small model configurations shared by tests

tests/
├── integration/    application checks across installed packages
├── tools/          planning, Plan measurement and diagnostic-tool tests
├── qualification/  gate runner and acceptance-rule tests
└── repository/     packaging, naming, documentation and boundary checks
```

Inside `engine_tests/`, the mirror goes all the way down: a test for
`packages/engine/src/shadowspill/profiling/` lives in
`packages/engine/tests/engine_tests/profiling/`. A test that exercises two packages
belongs to the one it is about, not the one it imports for a fixture.

Independent PyTorch training and checkpoint checks live in
`tests/qualification/reference/training/`; their implementation belongs to `qualification/reference/training/`.

`packages/engine/tests/engine_tests/native/` mirrors `packages/engine/csrc/` the same way: `runtime/`, `simulator/`, `backends/`,
`adapter/`, `network/` and `planner/search/algorithms/pressurefit/` hold the
canaries for those libraries. At its root, the ABI sizes and build canaries are
about the build as a whole, the SSD canary covers the SSD extension, and
`test_canaries.py` runs the canaries through `ctest` from the suite, the
`network` set only when `SHADOWSPILL_NETWORK_PEER` names a peer.

Tests requiring the execution allocator run in fresh workers. Build/profile stay
in ordinary PyTorch processes; each execution worker installs the allocator before
creating device tensors. Shared helpers under `packages/engine/tests/engine_tests/execution/`
exercise this boundary and keep process setup out of individual cases.

The default options deselect two markers. `fresh_process` tests assert what
their own process's allocator looks like, so CTest runs each in a process of
its own, as listed in `packages/engine/tests/engine_tests/integration/CMakeLists.txt`. `slow` tests are end-to-end
service and phase integration runs of minutes each; `pytest -m slow` runs them.

Long-running numerical and throughput acceptance belongs in `qualification/`,
not in the ordinary unit suite.
