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

/* dlopen each path, resolve the descriptor symbol, and check the descriptor
   against this build's ABI. On failure nothing stays loaded. The array is
   caller-owned and freed with free(); `count` may be zero, which loads
   nothing and succeeds. */
ShadowSpillStatus shadowspill_pytorch_libraries_load(
    const char *const *paths,
    uint32_t count,
    ShadowSpillPytorchLoadedLibrary **loaded
);

/* Close the first `count` libraries, in reverse. Does not free the array. */
void shadowspill_pytorch_libraries_unload(
    ShadowSpillPytorchLoadedLibrary *loaded, uint32_t count
);

/* Flatten the loaded libraries' entries into the two lists a runtime config
   carries. Both are caller-owned, freed with free(), and needed only for the
   create call -- the runtime copies them. */
ShadowSpillStatus shadowspill_pytorch_library_entries(
    const ShadowSpillPytorchLoadedLibrary *loaded,
    uint32_t count,
    ShadowSpillPoolMemoryDescription **pool_memory,
    uint32_t *pool_memory_count,
    ShadowSpillLaneDescription **lanes,
    uint32_t *lane_count
);

/* Registered with on_exit by bootstrap; closes without waiting. */
void shadowspill_pytorch_process_exit(int status, void *argument);

#endif
