# Shared client application code

These helpers consume an already initialized training client. They import no
engine, model catalog or accelerator framework. Setup owns model configuration,
initialization, preparation and worker acquisition.

[loop.py](loop.py) provides one `update()` function for whole `step()` calls or
split `forward_backward()` contributions followed by `optim_step()`. The caller
chooses optimizer controls and optional final gradient scaling. Scaling is applied
once at the optimizer boundary, and the returned loss uses the same scale.
Omitted controls retain the client's defaults. Qwen, Llama and regression all
use this function.

The same module contains the fixed-window, globally normalized training loop
used by the Qwen applications. It handles schedules, held-out evaluation,
metrics and checkpoint cadence using settings and data, without model-specific
branches. Applications with different evaluation or data-normalization policies
can compose the shared update helper in their own loop.

[records.py](records.py) serializes application settings and data identities and
publishes a checkpoint reference after the server confirms the checkpoint commit.
The same helper writes Qwen, Llama and regression checkpoint records. Model-specific
setup validates its settings and source before restoring that checkpoint.

The application owns loop policy; the installed engine remains the sole owner
of execution, accumulation, optimizer updates and checkpoint state.
