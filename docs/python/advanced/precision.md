# Parameter precision

Models supply their usual precision defaults. Advanced callers can override
individual tensor dtypes during setup:

```python
from shadowspill_client import Model

model = Model(
    "qwen35",
    config={"preset": "9b"},
    precision={
        "parameter_dtypes": {
            "*.mixer.A_log": "float32",
            "*.mixer.dt_bias": "float32",
        },
    },
)
```

These two FP32 rules are already defaults for dense and MoE Qwen3.5; ordinary
callers can omit them. A user rule replaces the default for each matched tensor.
For example, choosing `"bfloat16"` explicitly changes its storage when the model's
operators support that combination. Other tensors retain their model defaults.

## Matching and scope

Keys are full tensor names or shell-style globs such as `blocks.*.mixer.A_log`.
They select floating parameters and floating registered buffers. Integer buffers
remain unchanged. Supported storage dtypes are `float16`, `bfloat16`, `float32`
and `float64`; quantized storage uses its provider's configuration.

User rules take precedence over model defaults. Within either mapping, an exact
name takes precedence over a glob. Conflicting matches, including conflicting
names for a tied parameter, are errors; dictionary order has no effect. Each user
pattern must match a floating tensor when setup constructs the model on meta.
This catches misspellings before capture. Selecting one alias selects the shared
tensor, and dtype selection preserves trainable/frozen flags and LoRA targeting.

The service applies this policy after constructing the model and adapters, before
loading initial weights and capturing its computations. It records the user rules
in resolved settings and pins the model-default declaration through its source
identity. Saved Plans and checkpoints retain their declared tensor dtypes.

This controls **stored model tensors**. Activation arithmetic, gradient
accumulation and optimizer state have their own settings. A model must implement
the required mixed-dtype arithmetic; the map does not insert automatic casts into
its forward function. Unsupported combinations fail preparation.

## Defaults for a custom model

Add `parameter_dtypes` to its `ModelComponent` registration, for example:

```python
ModelComponent(
    "my_project/model@1",
    Config,
    create,
    inputs,
    parameter_dtypes={"router.weight": "float32"},
)
```

No custom base class or `configure` hook is needed for this option. The engine
handles it before passing other precision options to an optional hook. Default
patterns may describe layers absent from a particular architecture configuration.
Include this declaration in the component's source digest, as described in the
[extension guide](writing-extensions.md).

For direct PyTorch construction outside the service, use
`shadowspill.pytorch.precision.apply_parameter_dtypes(model, defaults=..., overrides=...)`
before loading weights or building an optimizer. It is a setup operation, not a
replacement for `Module.to()`. A later blanket dtype cast changes ordinary module
storage, and converting back to FP32 cannot recover values already rounded to BF16.
