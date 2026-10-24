# Public pipeline refactor

Status: implementation underway. CPU-only artifacts/planning/reporting and the
single-GPU build/profile/Executor pipeline work for qualified dense models.
Caller migration, provider qualification, distributed support and legacy cleanup
remain. See [progress](progress.md) for the latest checks and limitations.

Development host: Chicago, using its `codex` tmux session.
Design checkout: `/home/shein/Documents/grad_school/research/shadowspill-offline-pipeline-refactor`.
Branch: `codex/offline-pipeline-refactor`, based on clean `master` at `bec7950b`.
The three reviewed Chicago commits were pushed to `origin/master` on 2026-10-24
with explicit user authorization. MLOps is clean at `0274515`.
These design notes remain the active version; see [progress](progress.md).

The [approved implementation order](implementation-order.md) puts working CPU-only
planning first, then the single-GPU pipeline, and distributed execution/sharding
last. The user approved autonomous work through this order.

Start with the [consolidated public API review](public-api-review.md) for the
current common workflow, including execution and gradient defaults. The detailed
reviews below retain rationale and protocol questions. Substantive new public
design decisions still require review; implementation follows the approved design.

This is a library-wide refactor. Quickstart and Trainer will compose the public
phases, but do not define their boundaries. The user requires agreement on public
functions, classes, and responsibilities before architectural changes are made.

## User requirements

1. Export, compile, and lower a model into an unprofiled ShadowSpill program
   independently of runtime pools. Prefer offline construction, but the user
   permits build to require a GPU when compiler/provider preparation needs one.
   Build finalizes task definitions; exact preparation hooks remain to specify.
   CPU-only planning is still required.
2. Obtain task costs through GPU measurements, supplied external measurements,
   or explicitly supplied estimates. GPU measurement in the current process must
   be optional. Require complete supplied coverage; do not silently invent
   missing task durations or memory requirements.
3. Plan and search on CPU with execution/spill budgets and configurable host
   transfers and collective communication. Collective costs may be measured or
   estimated using scale-up/scale-out bandwidths and topology.
4. Execute a selected plan with the GPU runtime.
5. Give each phase a clear public API and a persistent output usable in another
   process. Keep responsibilities and source layout easy to explain.
6. Rethink artifact persistence and caching as part of this design. Preserve the
   separation of Inductor/Triton caches from portable experiment artifacts.
7. Finish with repository cleanup: identify redundant implementations, migrate
   callers, and remove superseded APIs/code and stale documentation/tests. The
   user selected removal after migration rather than deprecated public wrappers;
   preserve existing capabilities under their new owners.
8. Keep the common build calls simple through defaults, while exposing partition
   options, recomputation behavior, and parallelism/sharding controls for advanced
   callers. Internal compiler passes need not be public to expose their policies.
9. Preserve connections to PyTorch module paths, parameter/buffer names, and
   available value/source names through stages and planning artifacts. Reports
   and traces should explain the model computation behind generated identifiers.
10. Build, profile application, planning, and search must not require a ShadowSpill
    runtime, its pools, or installation of its pluggable allocator. Measurement
    should also be independent of them, but the user explicitly allows
    allocator-backed profiling if necessary to retain admission/correctness
    information. Do not trade correctness for pool independence. Planning
    describes capacities and placements without allocating real pools.
11. Execution uses a separate fresh process after GPU compilation/profiling. The
    execution process installs the allocator before CUDA initialization and owns
    the runtime pools. Expose a worker API with an optional convenience launcher.
12. Planning/search should produce a persistent full execution plan that all
    participating ranks can load. Bind the current rank, device, and real groups
    at execution, with the plan serving as an executable artifact for its declared
    program and parallelism configuration.
13. The central parallelism declaration and shared plan must be extensible to
    other models and techniques. EP and CP are examples, not a closed set of
    supported modes. The initial implementation assumes symmetric computation
    with ownership and data that may differ by logical rank. Execution workers
    can bind to arbitrary logical ranks through
    a consistent launch mapping; physical process/GPU numbering must not define
    the program's logical ownership.
14. Keep the initial build API to one model describing the common rank-local
    computation. Asymmetric per-rank modules/graphs and their coordinated
    planning are out of scope. Document this limitation rather than introduce
    factories or role-module collections now. If that capability is added later,
    the user tentatively prefers explicitly supplied modules over factories.

The recent GLM-5.3-Flash stage traces motivate an acceptance example. They are not
a reason to add a special GLM replay path before the general APIs are designed.

## Agreed decisions

The user selected the following choices on 2026-10-23:

| Decision | Agreement |
| --- | --- |
| Phase outputs | `Program → ProfiledProgram → Plan`, with separate reusable profile data. These names are provisional. |
| Build API granularity | Public `build_forward`, `build_forward_backward`, and `build_step` entry points sharing internal passes. The user's later request expands the earlier single-builder choice. Individual export/compilation/lowering passes remain internal initially. |
| Build completion | Build finalizes task implementations, layouts, and dependencies for the target, using a representative GPU where required. `measure_program` warms up and records timing/memory without changing those definitions. Missing dynamic memory evidence remains explicit for profiling; finalized tasks do not imply fully known workspace. Compatible library-internal lazy compilation is allowed. |
| Build control style | Named arguments with simple defaults and focused policy objects for advanced partitioning, recomputation, parallelism, and related controls. No mandatory all-in-one `BuildOptions` object. |
| Partition default | The user selected `partition="auto"` by default, retaining `"whole"` and a custom `PartitionPolicy` for explicit stage boundaries. Partitioning remains build work because it determines the compiled task boundaries. Automatic structure detection is a heuristic, not a promise of a good split for every model. |
| Operator recomputation | Use `operator_recompute` for the build argument. The user selected the existing heuristic plus custom operator-level retention policies. Keep current default behavior and let advanced callers influence which intermediate outputs are retained or regenerated during build. Planning independently selects among built variants. Exact policy protocol, fields, and interaction with replay/effect constraints remain to design. |
| Profiling boundary | `measure_program(...) → Profiles`; a separate CPU function `apply_profiles(program, profiles) → ProfiledProgram`. Function names are provisional. |
| Measurement input preparation | Prepare valid inputs automatically for supported tasks, with caller-provided values or generation logic for custom requirements and realistic distributions. Backward uses saved tensors from its corresponding forward. Unsupported semantics require caller guidance rather than arbitrary shape-only values. Exact override scope and data/callback signatures remain to design. |
| Measurement selection | Profile all stages by default; accept an explicit stage list to restrict a run. The user chose this simpler interface instead of selecting missing tasks through a new helper or automatically skipping entries from existing Profiles. Compatible internal task forms may still share measurements without leaving selected stages uncovered. Exact stage-selector representation remains to design. |
| Planning resource inputs | Use explicit `memory_budgets` and `communication_info` arguments. The user requested including topology inside `communication_info`. These describe execution/spill memory limits; host/device and inter-GPU bandwidths, latencies, and measured timings; and the placement/connectivity assumed for logical participants. Exact fields, defaults, measured-collective, and sweep schemas remain to design. Compute profiles stay associated with the GPU/kernel configuration they describe. |
| Memory budget scope | The common execution/spill byte pair applies per logical rank. `execution_bytes` is the total GPU allowance, including runtime/provider overhead; planning derives usable pool capacity after accounting for overhead and reservations. `spill_bytes` is a separate per-rank spill capacity, not a whole-process host-memory cap. Exact overhead inputs, per-resource overrides, and aggregate host-limit handling remain to design. |
| Communication cost organization | Combine topology, supplied rates, latencies, and optional measured transfer/collective timings in `communication_info`. Do not add separate topology or `transfer_measurements` planning arguments. Planning/search remain CPU-only and calibration data are reusable. Exact fields, matching, interpolation, and estimation algorithms remain to design. |
| Communication field shape | Use flat numeric fields with explicit units, such as `fetch_bytes_per_second` and `fetch_latency_seconds`, alongside topology and optional measured timings. Retain optional detailed topology overrides. Nested per-transfer rate/latency records were not selected. Exact measurement/link records and cost-model conventions still need to be finalized. |
| Communication calibration API | Expose public `measure_communication(...) -> CommunicationInfo`, optional and independent of `measure_program`. Users may supply/import topology, rates, and timings instead. Host-copy measurement can use one GPU; direct collective measurements require the participating group. Planning never triggers calibration. Exact arguments, probe selection, fields, and persistence remain to design. |
| Transfer cost precedence | Use a compatible measured timing first; estimate uncovered cases from supplied rates and topology/cost assumptions. If neither provides a cost, report the missing requirement instead of inventing rates or running GPU calibration. Preserve whether each selected cost was measured or estimated. Exact matching and curve handling remain to design. |
| Fallback collective model | When measured timing is unavailable, planning selects a built-in collective estimation model from topology by default, with an advanced explicit override. Use supplied rates/latencies and show assumptions. Exact algorithms, field/override types, latency defaults, traffic conventions, and contention models remain to design. |
| Topology organization | Describe node membership in the common API, with optional explicit link/shared-resource details. Do not require a complete hardware graph for ordinary configurations. Node labels describe planning placement; they do not define model replicas or logical communication groups. Exact fields, homogeneous assumptions, and advanced link schemas remain to design. |
| Memory evidence | `Program` holds static graph/storage structure; `Profiles` holds task durations and dynamic workspace/allocation observations. |
| Estimated memory boundary | Memory estimates are for CPU planning and simulated timelines only. A plan whose selected work relies on estimated memory is not executable; the worker rejects it. Execution requires complete non-estimated memory information from compiler/provider descriptions and profiling as needed. Replace estimates and plan again before execution; do not add worker-side measurement/replanning to turn estimates into executable requirements. Exact status/reporting fields remain to design. |
| Profile completeness | Every required entry needs a supplied value. The user clarified that manually supplied estimates are allowed, superseding the earlier measured-only choice. No implicit estimator fallback in `apply_profiles`; retain the distinction between measurements and estimates. |
| Task profile organization | Keep each compatible task's timing and missing memory information together in one record within `Profiles`. GPU measurements and manual/external values use the same organization and retain their sources; measured timing may coexist with estimated memory. Exact field names, context keys, and metadata remain to design. |
| Profile value representation | Use ordinary numeric fields for durations and memory sizes. The user rejected `Measured(...)` and `Estimated(...)` wrapper classes as unnecessary complexity. Keep any source/measurement classification in plain metadata; exact metadata fields and defaults remain to design. |
| Manual profile classification | Manually supplied values without measurement metadata default to estimates. GPU measurement supplies its own metadata, and external imports may supply it explicitly. This does not fill missing numbers or override compiler-known memory. Estimated memory remains simulation-only; estimated timing alone does not prevent execution when the required memory information is verified. Exact metadata fields and consistency checks remain to design. |
| Profile merging | Later supplied values overwrite conflicting earlier values by default for compatible task/context entries. Fill missing fields, preserve fields absent from the later record, and retain the source of the selected values. Do not implicitly average samples or select the fastest result. Overwriting does not waive compatibility checks. Exact method spelling remains provisional. |
| Profile collection scope | One hardware/runtime setup per `Profiles` collection. A collection may contain many tasks, shapes, variants, ranks, and measurement/import runs, and may be reused across compatible Programs. Select the setup through the collection passed to `apply_profiles`, without a multi-setup selection layer. Exact setup metadata and compatibility fields remain to design. |
| Public artifact mutability | `Program`, `ProfiledProgram`, and `Plan` expose inspectable but read-only structure. `Profiles` remains editable for supplied measurements, estimates, and overrides. `apply_profiles` captures the applied values independently, so later edits cannot alter an existing result. Timing-only changes require applying profiles and planning again, not rebuilding compatible task code. Exact accessor and implementation details remain to design. |
| Program scope | One concrete candidate per `Program`, with fixed input shapes and number of rank-local examples where applicable, containing its save/recompute alternatives. Sweeps compare multiple programs and reuse compatible profiles. |
| Planning API granularity | `plan_program` performs core planning/search for one candidate/budget. `sweep_programs` (formerly proposed as `search_programs`) is a thin, optional convenience over public calls for multiple cases and collected results. Users can write their own loops without losing core planning capability. Both are CPU-only and perform no build or measurement work. |
| Planning result and failure behavior | `plan_program` returns `Plan` directly on success and raises an exception with diagnostics when it cannot produce one. Preserve the distinction between infeasible constraints and search exhaustion. `sweep_programs` records expected unsuccessful cases and continues, retaining successful plans and failure explanations. Do not add a mandatory result wrapper to single-plan calls. Exact result fields remain to design. |
| Search result retention | Retain one resulting Plan for every successful candidate/budget case, plus a best-per-budget lookup. Retain status and diagnostics for failed cases. This permits inspecting alternatives without replanning; it does not require retaining every internal schedule explored while solving a case. Exact accessors, ranking, and payload sharing remain to design. |
| Plan inspection and reporting | Expose `Plan.summary`, `schedule`, `timeline`, `memory_layout`, and `diagnostics`. Use a separate CPU-only `write_report(plan_or_search_result, path)` function for plan reports and sweep comparisons. Rendering consumes returned or loaded data without repeating pipeline work. Artifact saving remains a separate explicit operation. Exact nested fields and report formats remain to design. |
| Forward/backward boundary | `build_forward_backward` accepts multiple microbatches and accumulates within the call. Existing gradients may be supplied; absent gradients are created at their first backward contribution. Final gradients remain persistent and no optimizer update runs. |
| Incoming-gradient execution modes | One forward/backward Program/Plan supports fresh gradients by default and accumulation into existing gradients through explicit `accumulate_gradients=True`. Build/profile/plan account for both behaviors, their memory, and required reductions. Already-reduced incoming contributions must not be reduced again. Presence/loading validation and internal representation remain to specify. |
| Parallelism declaration | All three builders accept one explicit declarative `parallelism` specification owning logical groups, weight/gradient/optimizer sharding, and advanced per-parameter overrides. Build resolves it once; measurement/execution bind real groups by name. |
| Simple DP usage | The common DP case must have a small configuration without manually declaring groups or naming every parameter. The user selected sharded weights, gradients, and optimizer data as its default, with masters following optimizer ownership. Advanced settings retain independent component controls. Exact shorthand spelling remains under review. |
| Component storage groups | For each parameter, weights, gradients, and optimizer data share one storage-sharding group by default, with optional component-specific group overrides. Master weights follow optimizer ownership. Differing layouts require explicit redistribution represented and costed in the program/plan. Exact fields remain to design. |
| Storage object organization | Use a flat object with one common group, per-component sharding modes, and optional weight/gradient/optimizer group overrides. Do not require a nested layout object for each component. Masters follow optimizer ownership, and omitted fields inherit from the explicit storage default. Exact spelling and omitted/reset representation remain to finalize. |
| Computation independent of parameters | Distributed computation must also describe input/activation layouts and participating operations independently of parameter placement. Context-parallel attention is the user's example: the attention computation on Q/K/V has no learned parameters of its own. The mechanism must remain generic rather than enumerate parallel techniques. |
| Parameter ownership entry point | Model-specific ownership is supplied explicitly inside `parallelism`, rather than discovered through a required method on the model. Reusable model helpers may construct the declarations. Build validates them against the model and resolves them with requested storage sharding; exact fields remain open. |
| Parameter declaration organization | Keep parameter placements and parameter-related storage in separate maps inside `parallelism`. Placements describe the logical parameter slice used by each local model; storage describes persistent weights, gradients, masters, and optimizer data. Input/activation layouts and distributed operations remain independent of both maps. Exact field names remain provisional. |
| Logical group representation | Explicit named, ordered groups of logical ranks are the core representation. An optional mesh helper generates those groups; a mesh is not mandatory. The user's later replica-based proposal scopes user declarations to one replica and generates the repeated groups. Exact constructors and group-reference fields remain to design. |
| Model-facing group references | Compatible layers/operators receive typed logical group references obtained from the central specification. Static metadata is available during CPU construction; measurement/execution later binds real process groups and provider resources. Exported artifacts retain portable identities/metadata. Exact type names, fields, and accessors remain to design. |
| Backend resource initialization | Save portable resource descriptions in the program/plan; compatible providers initialize live resources in measurement/execution workers. Advanced callers may supply compatible existing resources. Resource identities express sharing across operators/layers, and memory requirements remain explicit. This does not select a general per-operator registry. Exact description, initialization, cleanup, and binding schemas remain to design. |
| Core input boundary | Builders describe rank-local model inputs, and execution workers receive rank-local tensors. Optional helpers can split/distribute global batches. Distributed layouts remain explicit where needed; the core does not require global input payloads or own their distribution. |
| External input placements | Put distributed input-layout metadata in a separate map inside `parallelism`, keyed by paths into `example_args`/`example_kwargs`. Keep the example arguments as ordinary values rather than requiring per-tensor layout descriptors. Exact map/path syntax and layout propagation/validation remain to design. |
| Input placement composition | Accept either one `Sharded`/`Replicated` placement or a list of placements for a composed input layout. Do not require an `InputLayout` wrapper. The list describes the supplied local inputs, not an instruction to redistribute them. Exact advanced mapping and composition validation remain to design. |
| Input identity across groups | Distinct instances of an input's declared group and repeated complete-state replicas use independent logical inputs by default. Intentional sharing is explicit and independent of parameter sharing. Independence imposes no requirement that values differ; the core still receives caller-supplied rank-local payloads. A composed layout can declare sharing using `Replicated` alongside `Sharded`; access to sharing groups across repeated replicas remains to design. |
| Number of input examples | The common training call uses one rank-local input example plus `num_examples`, with an explicit per-example form for differing shapes. The user requested this order-neutral name instead of `accumulation_steps`. One example is a complete model/objective call and may itself contain a batch; it does not mean one dataset record or one optimizer update. Repeated slots accept independent runtime data; the count does not implicitly change objective normalization. Exact argument packaging remains to design. |
| Microbatch ordering boundary | The user requests choosing data/microbatch ordering during planning/search rather than build. Build retains reusable task variants and semantic dependencies; planning selects the traversal and binds gradient initialization/accumulation. One fixed input-shape/example-count candidate can support multiple orders without rebuilding or remeasuring its tasks. Exact planning controls remain to design. |
| Optimizer ordering boundary | The user selected moving optimizer ordering into planning/search as well. Build defines the optimizer computation and required dependencies; planning can choose stage-interleaved updates or updates at the end using the same compiled task forms. Gradient completion, reads of old weights, shared parameters, and global optimizer operations constrain legal placement. Exact policy fields/defaults remain to design. |
| Model-independent inputs | Example inputs must describe the complete model/objective call, including model-specific metadata. Do not require an LLM-specific schema or a universal `seqlen` field. Tensor geometry, runtime values, static arguments, and separate profiling metadata must be distinguished. Positional/keyword packaging is selected below; exact custom-container and metadata support remain to design. |
| Input-call syntax | The user selected separate `example_args` and `example_kwargs` on the builders, rather than a required `CallInputs` object. They describe the model/forward wrapper or training objective call and support nested arguments. Custom-container support and the advanced explicit per-example representation remain to design. |
| Concrete builder/parallelism shape | The user accepted the consolidated examples in `public-api-review.md`: `Parallelism.dp(ranks_per_replica=...)`, an advanced `Parallelism` with named groups and separate parameter-placement/storage defaults and overrides, and `parallelism.group(name)` returning typed references. Training builders call `objective(model, *example_args, **example_kwargs)`; forward calls the model with its own example arguments. Exact target/provider and remaining artifact/runtime details are separate reviews. |
| Replica meaning | A replica is a rank group that collectively owns the entire logical model and its associated training state. It may contain eight DP ranks processing different data while weights, gradients, and optimizer state are sharded across them. `num_replicas` counts repeated complete-state groups, not data-parallel workers. Gradient reductions can still be required inside a replica when `num_replicas=1`. |
| Repeated replica description | The user requests describing one complete replica, then supplying `num_replicas`; total participants are ranks per replica times replica count. Repeat the same groups, placements, and settings, keep sharding within a replica, and infer gradient reductions across corresponding parameter/shard copies between replicas. This replaces the earlier proposal requiring callers to enumerate outer copy groups. Exact API spelling and storage-sharding fields remain under review. |
| Global sharding through replica size | The user clarified that globally sharding a job is expressed by enlarging the replica to cover all participants and setting `num_replicas=1`. Smaller replicas repeated multiple times express hybrid sharding. No separate cross-replica sharding mechanism is needed for these cases. Replicas are logical units, not fixed physical-node boundaries. |
| Parameter selection | Use module-qualified parameter/buffer names and patterns, with defaults and named overrides resolved during build. Parameter-object references are not the selected public selector style. Exact pattern syntax and unmatched-selector behavior remain to design. |
| Name override precedence | Resolve each map independently of entry order. Exact names take priority over matching wildcard patterns; without an exact match, matching patterns must agree or build reports a conflict. Use the explicit default when no entry matches. Do not infer wildcard specificity or use last-match precedence. |
| Tied parameter names | Accept any registered name of the same Parameter. Resolve declarations over that shared identity using the agreed precedence, apply the result to every use, and reject conflicts at the winning priority. Preserve all source names and shared parameter/gradient/optimizer ownership. Equal-valued independent parameters and arbitrary storage-sharing views are not automatically merged. |
| Storage-default inheritance | Select the parameter-storage entry using the agreed name precedence, then fill its omitted fields from the explicit storage default. Do not merge matching wildcard entries or inherit a wildcard entry into an exact-name entry. The resolved configuration must be complete. |
| Placement terminology | Use `Sharded` for tensor placement; reserve partitioning terminology for computational stage boundaries. The user rejected `StateContract` and `StateRule` as public API names. Container and field names remain provisional. Model sharding and ShadowSpill storage sharding must remain distinguishable. |
| Parameter shard mapping | `Sharded(group=..., dim=...)` defaults to equal contiguous slices in the group's logical order. For local extent L and group size G, position p represents indices [pL, (p+1)L) of global extent GL. Allow explicit mappings for other assignments within the initial common-local-shape/graph scope. This declares the model's ownership; it does not rewrite its computation. Exact advanced mapping fields remain to design. |
| Gradient reduction inference | Infer SUM reductions for trainable parameters from their placements, with optional per-parameter overrides for gradients already combined by model backward. The criterion is copies of the same logical parameter or shard, including copies across different groups doing model sharding. `Sharded` alone does not imply no reduction. Frozen parameters need no parameter-gradient reduction; objective normalization remains separate. |
| Gradient completion | Distributed `build_forward_backward` returns reduced gradients in the declared sharded/replicated layout by default, with explicit deferred synchronization available. |
| Static workspace | Build derives compiler-visible workspace sizes/lifetimes into `Program` and explicitly identifies unresolved custom-op memory. `Profiles` supplies timings and missing/dynamic memory information. Unknown scratch is not silently zero. |
| Source attribution | Preserve internal IDs and attach semantic source metadata, including module paths, parameter names, and available graph/source locations. Display semantic names in reports and traces. Exact fields remain to design. |
| Runtime ownership | Build, profile application, planning, and search are independent of the runtime, allocator installation, and pools. Measurement should also be independent; the user's later clarification allows allocator-backed profiling if needed for admission/correctness. No fallback backend has been selected. |
| Execution process | Execution uses a separate fresh process with allocator installation before CUDA initialization. Expose an explicit worker API usable with caller-created workers/torchrun, plus an optional launcher. Workers persist across training steps. |
| Trainer and Executor | Trainer is the normal training interface and uses the public Executor API. Expose Executor independently for fine-grained control of a prepared plan; do not create a separate private execution path for Trainer. A worker is the process hosting an executor, rather than another required public execution object. Exact signatures, resource lifecycle, and Trainer orchestration remain to design. |
| Executor setup | Creating an Executor normally handles allocator/pool setup and resolves the Plan's logical groups/provider resources. Retain explicit compatible runtime/resource overrides for advanced use. Trainer uses the same entry point in a fresh execution process. Tensor initialization and calls remain explicit; startup performs no build, profiling, or replanning. Exact signatures and binding schemas remain to design. |
| Executor rank binding | Supply the same complete launcher-worker-to-logical-rank mapping to every worker. Each Executor derives its logical rank from its launcher identity and this map. Coordinated startup validates identical maps, one-to-one coverage, shared Plan identity, and compatible placement before GPU collective execution. Preserve logical collective member order. This supersedes per-worker logical-rank declarations; exact rendezvous/context integration remains to specify. |
| Tensor initialization | Use a loader/initializer that fills Executor-provided destination buffers. Requests cover bounded chunks of the declared owned tensor regions; destinations are direct storage views where accessible or bounded staging buffers. Trainer helpers use this same interface. Exact request fields, callback protocol, and checkpoint/random helper APIs remain to design. |
| Runtime input delivery | `Executor.run` accepts sequences of per-example positional arguments and keyword mappings, with an optional indexed input loader for streaming. The Plan determines execution order; an example index identifies a stable input slot. Trainer uses this same public path. Exact loader lifetime, staging, output, and hyperparameter interfaces remain to design. |
| Invocation completion | `Executor.run` returns after dispatch without an unconditional final GPU-completion wait; callers can explicitly synchronize the executor's outstanding work. Preserve stream dependencies, tensor lifetimes, and readiness of CPU-visible results. This does not authorize concurrent invocations or an unbounded queue of steps. |
| Invocation result handle | `Executor.run` returns a per-call result handle, reusing the existing `InvocationResult` concept. `result()` waits for that invocation, including required writebacks, then returns its caller-owned outputs or propagates execution errors. Keep `executor.synchronize()` for outstanding executor work. This is not a device-wide wait or authorization for concurrent updates. |
| Subsequent invocation admission | Allow one active invocation per Executor initially. A subsequent `run` automatically waits for the prior invocation, including required writebacks, before reusing its working storage or dispatching new work. The caller need not explicitly retrieve the earlier outputs. Preserve completed handles and caller-owned results; retained storage still counts toward capacity. Do not introduce an unbounded queue or overlapping invocations. |
| Invocation result payload | Forward plans return the model's normal tensor/dict/tuple/nested output through the handle. Forward/backward and full-training plans return a small record with per-example losses and auxiliary metrics in example-slot order. Keep rank-local values and deliberate aggregation by Trainer/callers. Gradients remain Executor-owned. A common wrapper around every plan kind was not selected. Exact training record name and optional diagnostic/observation fields remain for the final API review. |
| Returned output lifetime | Returned computation results are caller-owned and remain valid across later calls. Transfer ownership safely or copy when necessary; retaining results consumes memory and must be accounted for. This does not automatically snapshot persistent parameters, gradients, or optimizer tensors. Exact result schema, storage location, and admission/release mechanics remain to design. |
| Runtime hyperparameters | Pass runtime-changeable settings through a plain `hyperparams` mapping on each `Executor.run`; Trainer supplies scheduler values through the same API. The saved Program/Plan identifies supported runtime values. A persistent setter is not the common path. Exact declarations, keys, optimizer-group overrides, omission/default behavior, and checkpoint handling remain to design. |
| Forward/backward gradient default | Each `build_forward_backward` invocation starts fresh gradients by default (`accumulate_gradients=False`); explicit `accumulate_gradients=True` adds new contributions to stored gradients. Accumulation across examples within the invocation is unchanged. Cover both behaviors in one Plan and do not reduce previously reduced incoming gradients again. |
| Persistent tensor export | Executor provides bounded, format-independent tensor export. Separate checkpoint helpers handle formats, file organization, and distributed publication; Trainer uses the same path. Keep executable artifact saving separate from live model/checkpoint values. Exact chunk protocol, metadata, checkpoint contents, and consistency/publication details remain to design. |
| Artifact persistence | Explicit save/load on `Program`, `Profiles`, `ProfiledProgram`, and `Plan`. An optional cache accelerates work but is not required for phase handoff. File format and dependency packaging remain open. |
| Artifact container | One archive file per saved phase artifact, containing readable metadata and required payloads. The user selected the archive over a small directory of files. Keep compiler caches and generated reports separate. Exact container format, metadata schema, extension, and compression remain to design. |
| ProfiledProgram bundle | `ProfiledProgram.save(path)` includes its Program and applied profile data, so loading does not depend on the original Program/Profiles artifact folders or their cache. Keep model checkpoints, compatible libraries, and disposable compiler caches external. Exact file format and deduplication remain to design. |
| Executable plan bundle | `Plan.save(path)` packages selected task code/graphs, schedule, memory layout, logical rank/group declarations, and input/state contracts so every rank loads the same artifact. Checkpoint weights and compatible runtime/provider libraries remain external. The plan targets a fixed parallelism configuration. |
| Initial executable software compatibility | Require exact recorded versions/build identities for relevant compiler, framework, runtime, and provider dependencies that affect the selected executable tasks. Include relevant local implementation identities rather than relying only on version strings. CPU archive inspection/reporting stays independent. This is not an exact match of all installed packages, paths, or GPU ordinals; target/driver/capacity requirements remain separately checked. Supported version ranges were not selected initially. |
| Public API and source organization | Use a small top-level `shadowspill` API with implementations organized by build, profiling, planning, execution, and supporting responsibilities. Keep CPU imports independent of GPU initialization/provider loading. Files/functions should be small and focused; use hierarchy within phase directories when it clarifies real components. Avoid unnecessary facade layers and duplicate data-object hierarchies. |
| Legacy API migration | Migrate repository callers, then remove superseded public APIs and duplicate implementations. Do not retain deprecated public compatibility wrappers. Temporary internal migration adapters are removed after callers move; preserve existing capabilities under the new owners. |
| Extensible parallelism and rank binding | Use generic contracts for models/techniques and their combinations; initially support symmetric computation with rank-dependent ownership/data. EP and CP are examples, not an enumeration of allowed schemes. Workers may bind independently of physical rank/device numbering. Exact logical layout, group, extension, and binding schemas remain to be agreed. |
| Initial symmetry scope | One model and a common rank-local computation/schedule. No initial model-factory or per-role module-collection API; asymmetric programs are explicitly unsupported. Explicit modules are the user's tentative preference if asymmetric support is designed later. |
| Distributed model contracts | Non-DP computation needs explicit contracts. Exact participation, placement, communication, resource, and binding interfaces remain to agree on; the existing EP path is the concrete case under review. |
| Model author guidance | Define a concrete, provider-independent authoring contract for the refactored system. Current recipes and MLOps behavior are examples and may change; compatibility must not depend on preserving their existing interfaces. |

The table records selected public names and behavior. Complete function
signatures, nested field schemas, exact archive format, and detailed source
mapping still need review; phase ownership and migration/removal policy are selected.

## Pending choices

See the [API draft](api-draft.md) for a concrete proposal. Its exact signatures,
remaining field names/defaults, file format, and runtime binding still need
review; the choices recorded above are selected.

Named build arguments with focused policies, explicit save/load with an optional
cache, portable executable Plan bundles, and ProfiledProgram bundles containing
their Program/applied profiles are now agreed. Saved phase artifacts use one
archive file. Exact container/schema and the organization/loading of executable
payloads remain open.

## Six remaining design reviews

The phase boundaries, main function names, replica meaning, parameter/input
placement organization, and provider-initialized resource descriptions above are
settled. Consolidate the remaining work into six reviews of concrete examples
and interfaces, rather than a separate preference poll for every field. These
are six review packages, not a promise that exactly six binary questions remain;
an example may reveal a substantive follow-up. Do not reopen selected behavior or
silently decide the remaining public design. Present a complete proposal for each
review and distinguish unresolved choices from implementation details.

| Order | Area | Concrete result still needed |
| --- | --- | --- |
| 1 | Complete public API and parallelism examples | Consolidate the three builder signatures, focused policies, Program/task/profile inspection, and the agreed parallelism behavior into complete simple-DP, repeated-replica, and mixed-sharding examples. Include a manual-profile CPU-planning example and concrete Plan/sweep result access. Finalize fields/accessors, buffer handling, and explicit shard mappings in context. Existing defaults, precedence, ties, ownership/storage separation, and input composition are already selected. |
| 2 | Build and measurement completion; model/provider obligations | Define when GPU-assisted preparation occurs, what a completed Program promises, and how required memory evidence is supplied. Specify the minimal provider resource/binding/replay/allocation interface using existing PyTorch metadata where possible. Preserve one-GPU local DP profiling, and distinguish simulation-only memory from executable requirements. Allocator-observer feasibility is an implementation investigation, not a preference question. |
| 3 | Communication and resource inputs | Finalize `CommunicationInfo` and memory-budget fields in one example: topology, rates/latencies, measurements, total GPU allowance and overhead. State byte conventions, collective estimation assumptions, and how provider-internal communication avoids double counting. Complete the optional calibration call and planning controls for ordering. Small-collective optimization and expanded asynchronous scheduling remain deferred. |
| 4 | Executor interface | Show concrete load/bind/initialize/run/result/export calls in a fresh worker: arbitrary consistent logical-rank binding, bounded initialization, input delivery, fresh/accumulated gradients, hyperparameters, completion, and compatibility/allocation mismatch handling. Trainer composes this same API. The already selected ownership and lifetime behavior is not reopened. |
| 5 | Artifact portability and compatibility | Specify contents and version/target/provider checks for the single-archive format, independently loadable Program/ProfiledProgram/Plan artifacts, selected executable dependencies, and optional caches. Explain what works after compiler caches are removed and what happens when the worker environment is incompatible. No worker-side replanning to repair estimated memory. |
| 6 | Source tree and migration | Phase-based layout, top-level API, and removal of old APIs after caller migration are selected. Complete the current-to-new capability map; migrate trainer, quickstart, recipes, benchmarks, tests, and docs; then remove redundant code and stale documentation. |

The first review is now presented as [concrete public API examples](public-api-review.md):
simple sharded DP, the three computation boundaries, measurement/CPU planning,
manual profile access, and the sixteen-rank mixed-sharding case. Its proposed
builder/parallelism shape has been accepted after explaining args/kwargs and the
objective call. Additional record schemas remain to finalize in their respective
reviews. The user also accepted [build finalization before measurement](build-completion-review.md).
The user selected flat rate/latency fields for [communication and resource inputs](communication-review.md);
detailed measurement/link schemas and cost-model conventions remain to finalize.
The [Executor interface](executor-review.md) review is in progress. The user
selected a per-call result handle; its previously selected lifecycle and output
ownership are retained. A complete rank map supplied to every worker is selected.
Native forward outputs and a training result record are selected. A subsequent
`run` waits for the previous invocation automatically. Exact relevant software
builds are selected for [executable compatibility](artifact-review.md). The next
grouped review, [source layout and migration](source-layout-review.md), now has an
accepted phase-based layout and top-level API. The user also selected migration
of repository callers followed by removal of old APIs and duplicate implementations,
without deprecated public wrappers. Work will use the separate
`codex/offline-pipeline-refactor` branch.

Migration/removal and the fresh-gradient default are now selected. Check
complete end-to-end examples against all accepted decisions, including remaining
record/default/provider/archive details from earlier reviews. Moving on does not
silently settle those details. This inventory is not a fixed count of dropdowns.
Allocator-observation and provider implementation investigations are engineering
work; do not turn them into unnecessary user preference questions.

The [implementation readiness note](implementation-readiness.md) separates the
remaining user choices from the final API consistency pass and engineering work.

The user explicitly deferred small-collective optimizations until after the
refactor. Communication batching/bucketing and avoiding eager post-update weight
gathers are follow-up work, not requirements for completing this design or its
initial implementation. The API draft retains the investigation as deferred
notes; do not expand the current API or planning work to implement it.

Current discussion: topology is included in `communication_info`; node-membership
organization and measured-first transfer precedence remain selected. Single-plan failure behavior, Plan inspection/reporting, and the
simulation-only boundary for estimated memory are also selected. ProfiledProgram
includes its inputs when saved, and each artifact is one archive file. Executor
setup ownership, initialization into supplied buffers, runtime sequences with
an optional indexed loader, asynchronous dispatch with a per-call result handle
and explicit completion wait, caller-owned outputs valid across calls, and per-call hyperparameter
mappings are selected. One forward/backward Program/Plan supports both fresh and
accumulating calls. Executor provides bounded tensor export with separate
checkpoint helpers. Common memory budgets are per rank and execution includes
runtime/provider overhead. Collective estimation has an automatic default with an
explicit override. Build now owns task finalization and GPU-dependent preparation
when needed; its exact hooks remain to specify. For symmetric pure DP, one representative
GPU can prepare/profile local task forms; directly measuring collectives requires
their participating ranks. The user proposes that optimizer tasks consume reduced
gradient shards, owned masters/weights, and optimizer tensors, with DP
communication outside the update. The user prioritizes local DP profiling without
live process groups and questions expanding the scope into asynchronous overlap.
The current recommendation is to start with ordered communication boundaries;
independent collective overlap can follow later. This also supports
forward/backward-only calls and local optimizer profiling; exact representation
and initial execution policy remain under review. Remaining resource fields, runtime arguments, and result/export protocols need
examples. Review the remaining
build/parallelism/object fields through complete end-to-end examples. Finish with
the source layout, caller migration, and removal of redundant APIs/files. Keep
the motivating GLM import-and-CPU-simulation example as the acceptance case.
This order guides design review and does not authorize unapproved API choices.

Independent public communication calibration is now selected as
`measure_communication(...) -> CommunicationInfo`, alongside manual/imported
information. Built Program/ProfiledProgram/Plan structure is now selected as
inspectable but read-only, with editable Profiles and independently captured
applied values. Manually supplied profile values without measurement metadata now
default to estimates; exact metadata fields and memory-evidence validation remain
to design. SearchResult now retains each successful candidate/budget case's Plan
and provides a best-per-budget lookup. The user accepted keeping `sweep_programs`
as a thin, optional helper: core planning/search belongs in `plan_program`, and
callers can loop over it themselves. The helper aggregates cases and failures;
compatible-plan reuse should remain accessible through core planning inputs.
`SearchResult` remains a provisional object name; no additional naming decision
is implied by renaming the helper. The pending proposal for optional work counts
and time/throughput ranking remains unapproved and belongs in the complete API
review rather than another isolated poll. The user asks how much design remains;
the six review packages above consolidate the earlier nine-area inventory.

The first item should be checked against three examples before moving on:

- One eight-rank replica doing DP with sharded weights, gradients, and optimizer
  state; its ranks can process different data and still reduce gradients inside
  that replica.
- Two copies of that same replica, with matching gradient shards synchronized
  across replicas.
- A symmetric distributed model using named groups and different model shards
  within each replica, without hard-coding a closed list of parallel techniques.

The user supplied a concrete version of the last case: [one sixteen-rank
replica containing two EP8 groups](build-and-parallelism.md#required-example-two-ep8-groups-sharing-storage-inside-one-16-rank-replica),
with expert weights storage-sharded across corresponding pairs such as `(0,8)`
and attention weights storage-sharded across all sixteen ranks. Treat this as a
required expressiveness example when reviewing the parameter/sharding fields.
It does not choose new field names or gradient/optimizer storage defaults.

The user also raised non-contiguous input ownership as an expressiveness case:
split a sequence into eight chunks and assign ordered pairs `(0,7)`, `(1,6)`,
`(2,5)`, `(3,4)` to four logical group members. The design should describe these
global positions even when each rank packs its two chunks into one contiguous
local tensor. An optional ordered chunk mapping is proposed in the API draft;
its exact representation is not yet approved. Input metadata alone does not
implement the distributed attention algorithm or supply its runtime positions.

### Implementation investigations, not additional user preference questions

- Verify which compiler/provider paths can build offline and extract temporary
  allocation sizes, lifetimes, reuse, and alignment; identify explicit gaps.
- Qualify native allocation observation against the information required by
  execution. Use the user-permitted allocator-backed measurement fallback if
  needed; build and planning remain pool-independent either way.
- Verify that source attribution and task identities survive graph transforms,
  differentiation, and repeated equivalent layers without preventing reuse.
- Test save/load and execution in a fresh process with a different valid physical
  rank mapping, and with the original compiler-cache directory unavailable.

These investigations must substantiate the selected public behavior. If a result
would require changing that behavior, bring the concrete tradeoff back for review.
They do not reopen the agreed symmetric scope or add GPU requirements to CPU
planning.

## Acceptance example to work toward

After the general interfaces are agreed and implemented, a caller should be able
to load a previously built Program, or build offline where supported, supply the available GLM stage
measurements plus explicit values for all missing task costs/memory requirements,
specify memory budgets, transfer bandwidths, and any required topology/cost
information, and obtain a simulated timeline
entirely on CPU. Missing values must be reported instead of silently estimated.
External measurement provenance and explicitly supplied estimates must remain
distinguishable. Estimates are suitable for simulation; they do not constitute
observed runtime allocation evidence.

If the selected compiler/provider requires GPU preparation, producing the Program
may use a representative GPU first. Subsequent profile application, planning,
search, and timeline generation still run entirely on CPU. The proposed policy
for returning Programs with unresolved backend choices has not been selected.

The existing 40 traces cover five unique stages at batch sizes 1/2/4/8, both save
and recompute, at sequence length 8192. They omit the embedding-containing first
stage and optimizer work. Their memory peaks are not the per-task allocation
contracts used by the runtime. These limitations should be expressible through
the general interface rather than hidden by this example.

## Audit and implementation status

- [Current source audit](current-state.md): existing boundaries, public APIs,
  profiling dependencies, collectives, and artifact storage.
- [Build controls and parallelism proposal](build-and-parallelism.md): current
  locations and proposed responsibilities, awaiting design agreement.
- [Distributed model contracts](distributed-contracts.md): how the current EP
  path works, its existing contracts, and the changes proposed for offline build
  and portable execution.
- [Model authoring contract](model-authoring-contract.md): proposed concrete
  construction and compatibility requirements for model/operator authors,
  independent of any existing recipe or provider.
- [Memory inputs and profiles](memory-and-profiles.md): required information and
  the distinction between simulation inputs and execution evidence.
- [API and object draft](api-draft.md): proposed signatures, object contents,
  control defaults, and persistence boundary for review.
- Library/runtime implementation is on the refactor branch; main stays clean.
- Small single-GPU numerical tests have run on Chicago. No Della reservations or
  full GLM/Qwen training campaigns have been restarted.
- The normal Trainer constructor is under review: consume a saved Plan (with an
  optional existing Executor), or retain model/prepare orchestration. The user
  has been asked; that public choice is not silently selected by implementation.
- Provider/distributed qualification and caller migration precede deleting the
  remaining legacy paths. The detailed review documents retain design history;
  the progress log and public architecture document describe implemented behavior.
