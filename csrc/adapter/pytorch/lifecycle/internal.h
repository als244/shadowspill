#ifndef SHADOWSPILL_PYTORCH_LIFECYCLE_INTERNAL_H
#define SHADOWSPILL_PYTORCH_LIFECYCLE_INTERNAL_H

/*
 * From a config to a published runtime, and back: loading the backend,
 * creating the runtime, the physical-memory ledger, close, and the
 * process-exit hook. The backend outlives the runtime it serves.
 */

#include "../internal.h"

/* dlopen the library at path, resolve the two contract symbols, create the
   backend for the device and validate its table. On failure nothing stays
   loaded. */
ShadowSpillStatus shadowspill_pytorch_backend_load(
    const char *path,
    int32_t device_ordinal,
    ShadowSpillPytorchLoadedBackend *loaded
);

/* Destroy the backend and close its library; a zeroed value is a no-op. */
void shadowspill_pytorch_backend_unload(
    ShadowSpillPytorchLoadedBackend *loaded
);

/* Registered with on_exit by bootstrap; closes without waiting. */
void shadowspill_pytorch_process_exit(int status, void *argument);

#endif
