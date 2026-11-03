"""Resolve explicit candidate inputs/controls and retain local OOM failures."""

import gc


def example_inputs(candidate):
    unknown = set(candidate) - {"args", "kwargs", "build_options"}
    if unknown:
        raise ValueError(
            f"unknown candidate entries {sorted(unknown)}; "
            "use args, kwargs and optional build_options"
        )
    args, kwargs = candidate.get("args", ()), candidate.get("kwargs", ())
    if not (args or kwargs) or (args and kwargs and len(args) != len(kwargs)):
        raise ValueError(
            "candidate needs equally sized, nonempty args/kwargs sequences"
        )
    return {key: candidate[key] for key in ("args", "kwargs") if key in candidate}


def build_options(setup, candidate):
    overrides = candidate.get("build_options", {})
    topology = {"parallelism", "resources"}.intersection(overrides)
    if topology:
        raise ValueError(
            f"candidate build_options cannot override {sorted(topology)}; "
            "declare the shared worker topology in factory build_options"
        )
    return {**setup.get("build_options", {}), **overrides}


def attempt(entry, phase, operation, device):
    import torch

    from shadowspill.pytorch.failures import device_exhausted, exception_chain

    try:
        return operation()
    except Exception as error:
        if not device_exhausted(error):
            raise
        entry["failure"] = {
            "type": "OutOfMemoryError",
            "phase": phase,
            "message": phase
            + ": "
            + "; ".join(str(item) for item in exception_chain(error)),
        }
    # Leave the exception handler first so its traceback no longer retains
    # the failed task's temporary tensors, streams and compiled functions.
    gc.collect()
    if torch.cuda.is_initialized():
        with torch.cuda.device(device):
            torch.cuda.empty_cache()
    print(f"{phase}: {entry['name']} failed: {entry['failure']['message']}", flush=True)
    return None
