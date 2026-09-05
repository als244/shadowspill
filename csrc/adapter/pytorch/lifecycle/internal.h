#ifndef SHADOWSPILL_PYTORCH_LIFECYCLE_INTERNAL_H
#define SHADOWSPILL_PYTORCH_LIFECYCLE_INTERNAL_H

/*
 * From a config to a published runtime, and back: loading the backend,
 * creating the runtime, the physical-memory ledger, close, and the
 * process-exit hook. The backend outlives the runtime it serves.
 */

#include "../internal.h"

/* Registered with on_exit by bootstrap; closes without waiting. */
void shadowspill_pytorch_process_exit(int status, void *argument);

#endif
