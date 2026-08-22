#ifndef SHADOWSPILL_RUNTIME_INTERNAL_COMPLETION_TRACKER_H
#define SHADOWSPILL_RUNTIME_INTERNAL_COMPLETION_TRACKER_H

#include <stdint.h>

typedef struct ShadowSpillRuntime ShadowSpillRuntime;
typedef struct ShadowSpillCompletionTracker ShadowSpillCompletionTracker;
typedef struct ShadowSpillEventLease ShadowSpillEventLease;

int shadowspill_completion_tracker_initialize(
    ShadowSpillCompletionTracker *tracker
);
void shadowspill_completion_tracker_destroy(
    ShadowSpillRuntime *runtime,
    ShadowSpillCompletionTracker *tracker
);
ShadowSpillRuntimeStatus shadowspill_completion_submit(
    ShadowSpillRuntime *runtime,
    ShadowSpillBackendStream stream,
    ShadowSpillEventLease *event,
    uint64_t object_id,
    uint64_t allocation_id
);
int shadowspill_completion_poll(
    ShadowSpillRuntime *runtime,
    uint64_t *next_poll_nanoseconds,
    uint64_t *failure_object_id,
    uint64_t *failure_allocation_id
);

#endif
