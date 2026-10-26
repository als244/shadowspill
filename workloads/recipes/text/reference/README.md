# PyTorch text reference

Select `"engine": "pytorch"` in the [text runner](../experiment/) configuration
to train with ordinary PyTorch. Use `"device": "cpu"` or `"device": "cuda:0"`
and `"compile": false` for eager execution; compilation defaults to `true`.
The model, objective and data configuration are shared with the phase runner.
Select one fixed `max_tokens_per_microbatch`; the reference does not search.

The files each own one part of the reference:

| File | Responsibility |
| --- | --- |
| `run.py` | Resolve configuration, initialize the model and choose fixed geometry. |
| `engine.py` | Forward/backward, gradient accumulation, optimizer update and evaluation. |
| `loop.py` | Schedules, logging, evaluation/checkpoint cadence and packing records. |
| `checkpoint.py` | Atomic model/optimizer and loop-state saving, resume and retention. |

The reference uses ordinary allocations. It does not depend on the old Trainer,
preparation backends or a ShadowSpill Plan. Shared precision helpers retain compute
weights, optional masters and separately accumulated gradients. Metric callbacks
remain eager. `parameter_norms_every` enables norms on selected updates; observations
are disabled by default. An explicit `parameter_metrics` callback defaults to every
update unless a cadence is supplied.

Checkpoints retain the PyTorch reference format:
`checkpoints/step_XXXXXXXX/{state.pt,loop.pt,manifest.json}`. They include the
optimizer, schedules, RNG and packing progress. `checkpoint_weights="compute"`
chooses compute weights instead of masters; `checkpoint_include_optimizer=false`
omits optimizer state and resumes with a fresh optimizer. Checkpoints from the
saved-Plan runner are separate `.ssp` archives.
