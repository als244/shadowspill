# Validation on Chicago

Run these from the refactor worktree. Chicago's existing editable installation
points at the main checkout, so `python -S` plus explicit paths ensures these
checks import the branch under test. Other installations can use their ordinary
environment once it points at this worktree.

```bash
SS_PYTHON=/home/shein/miniconda3/envs/shadowspill/bin/python
SS_SITE=/home/shein/miniconda3/envs/shadowspill/lib/python3.12/site-packages
export PYTHONPATH="$PWD/src:$PWD:$SS_SITE"
export SHADOWSPILL_LIBRARY_DIRECTORY="$PWD/build/offline"

CUDA_VISIBLE_DEVICES='' "$SS_PYTHON" -S -m pytest \
  tests/shadowspill/offline tests/shadowspill/ir \
  tests/shadowspill/planner tests/shadowspill/simulator

CUDA_VISIBLE_DEVICES=0 "$SS_PYTHON" -S -m pytest \
  tests/shadowspill/build tests/shadowspill/profiling \
  tests/shadowspill/execution
```

`build/offline` must contain this branch's rebuilt native CUDA/PyTorch libraries.
Build them with the existing CMake setup before running the GPU checks. The
allocation observer and metadata-only CUDA storage require the new native code.

The GPU tests create separate execution workers. Those workers deliberately
block build, compiler, calibration and ArtifactStore entry points; several also
use empty compiler-cache directories. Numerical oracles cover repeated forward,
forward/backward and AdamW updates, tied parameters, fresh/accumulated gradients,
mixed precision, output ownership, bounded checkpoint restore and custom-op
registration. They establish these fixtures, not blanket provider qualification.

The CPU tests explicitly block framework/runtime imports while loading archives,
applying manual profiles, planning different traversals/budgets, and writing
reports. Estimated memory remains simulation-only. Corrupt artifacts, incomplete
profiles, allocation-contract mismatches and insufficient budgets are rejected.

Local generated reports and logs remain untracked in this directory. In
particular, `offline-demo/report/index.html` demonstrates CPU budget sweeps with
illustrative costs; it is not a GLM performance result. `forward-demo/report/`
comes from a built small-model Program. Current regression counts and limitations
are recorded in [the progress log](../progress.md).
