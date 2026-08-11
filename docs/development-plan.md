# Development plan

ShadowSpill is implemented in gated milestones:

1. tracked scaffold and frozen evidence;
2. immutable IR and serialization;
3. standalone simulator;
4. PressureFit and recomputation planning;
5. framework-neutral runtime with mock backend;
6. CUDA slab, PyTorch adapter, and physical budgets;
7. PyTorch lowering and structural profiling;
8. forward and training callables;
9. reduced-model numerical qualification;
10. full-model performance and release qualification.

Each milestone must pass its documented gate before the next becomes active.
Behavior corrections and structural changes are separate commits. The prior
repository is an external oracle only and is never imported by this package.

The detailed immutable starting plan and append-only engineering log are kept
under the locally ignored `docs/internal/` subtree.
