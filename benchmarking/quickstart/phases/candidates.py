"""Keep recoverable device exhaustion local to one sweep candidate."""

import gc

import torch

from shadowspill.pytorch.failures import device_exhausted, exception_chain


def attempt(entry, phase, operation, device):
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
