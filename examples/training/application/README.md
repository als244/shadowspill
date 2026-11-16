# Shared client application code

These helpers import no engine, model catalog or accelerator framework. Setup
and runtime are separate: setup returns a ready client; the loop consumes it.

[setup.py](setup.py) validates saved settings, input identity, progress and Plans
before resuming, creates the client and saves reconnect IDs before waiting for
initialization. A fresh run may supply an inspected setup preview. The service
owns preparation and worker initialization. The same lifecycle serves text and
non-text applications. Its cleanup context preserves the original error if
closing the client also fails.

[cli.py](cli.py) handles HTTP/SSH connection options and exact saved-settings
checks. Application entry points supply their configuration and data choices.

[Client parallelism](../../../packages/client/src/shadowspill_client/parallelism.py)
builds the existing DP/EP wire declarations from input placement, output rules
and expert parameter names. Basic DP infers input fields from schemas. The engine's
generic ownership interfaces remain unchanged; the client needs no engine import.

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
The same helper writes Qwen, Llama and regression checkpoint records. Shared
setup validates their settings and input identities before restoring a checkpoint.

The application owns loop policy; the installed engine remains the sole owner
of execution, accumulation, optimizer updates and checkpoint state.
