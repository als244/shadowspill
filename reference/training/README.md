# PyTorch text reference

Run an independent training reference with ordinary PyTorch:

```python
from reference.training.run import run_config

result = run_config("reference.json", ["device=cpu", "compile=false"])
```

The configuration names a model, objective, optimizer, initializer and data
source. Select one fixed `max_tokens_per_microbatch`; the reference does not
search. `device` selects CPU or an accelerator, and `compile` defaults to `true`.
The reference runs one worker and does not start a service.

The files each own one part of the reference:

| File | Responsibility |
| --- | --- |
| `config.py`, `configuration.py` | Resolve and validate reference configuration. |
| `text/` | Packed data, schedule arguments and logging used by this reference. |
| `run.py` | Resolve configuration, initialize the model and choose fixed geometry. |
| `initialization.py` | Fill the model through the configured `initialize` in bounded chunks, or reset its parameters from the seed when none is given. |
| `engine.py` | Forward/backward, gradient accumulation, schedules, optimizer update and evaluation. |
| `loop.py` | Logging, evaluation/checkpoint cadence and packing records. |
| `checkpoint.py` | Atomic model/optimizer and loop-state saving, resume and retention. |
| `state.py` | The reference's PyTorch model/optimizer checkpoint format. |
| `weights.py` | Compute weights, optimizer masters and accumulated gradients. |
| `hyperparameters.py` | Apply scheduled optimizer values and model-buffer controls. |
| `inputs.py` | Move reference inputs to the selected device. |

The reference uses ordinary allocations. It does not depend on the preparation phases or a ShadowSpill Plan. Its weight helper retains compute
weights, optional masters and separately accumulated gradients. Model materialization
and explicit objective scaling use shared `shadowspill.pytorch` helpers. Metric callbacks
remain eager. `parameter_norms_every` enables norms on selected updates; observations
are disabled by default. An explicit `parameter_metrics` callback defaults to every
update unless a cadence is supplied.

Checkpoints retain the PyTorch reference format:
`checkpoints/step_XXXXXXXX/{state.pt,loop.pt,manifest.json}`. They include the
optimizer, schedules, RNG and packing progress. `checkpoint_weights="compute"`
chooses compute weights instead of masters; `checkpoint_include_optimizer=false`
omits optimizer state and resumes with a fresh optimizer. Checkpoints from the
saved-Plan runner are separate `.ssp` archives.
