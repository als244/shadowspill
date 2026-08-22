#ifndef SHADOWSPILL_RUNTIME_TELEMETRY_INTERNAL_H
#define SHADOWSPILL_RUNTIME_TELEMETRY_INTERNAL_H

/*
 * The allocation-event ring, the trace ring, and the backend profiler.
 *
 * All three are diagnostic records. A step never depends on one: a full ring
 * stops recording, and a missing profiler callback is a no-op.
 */

#include <stdatomic.h>
#include <stdint.h>

#include <shadowspill/runtime.h>

#include "../memory/internal.h"

void shadowspill_append_allocation_event_locked(
    ShadowSpillRuntime *runtime,
    const ShadowSpillMemoryLease *allocation,
    ShadowSpillAllocationEventKind kind,
    ShadowSpillAllocationCategory category
);

void shadowspill_trace_append_enabled(
    ShadowSpillRuntime *runtime,
    ShadowSpillTraceEventKind kind,
    uint64_t task_id,
    uint64_t object_id,
    uint64_t allocation_id,
    uint64_t bytes,
    uint64_t detail_0,
    uint64_t detail_1
);



int shadowspill_profiler_is_valid(const ShadowSpillProfiler *profiler);

void shadowspill_profiler_set_enabled(
    const ShadowSpillProfiler *profiler, uint8_t enabled
);

void shadowspill_profiler_name_current_thread(
    const ShadowSpillProfiler *profiler, const char *name
);

void shadowspill_profiler_name_stream(
    const ShadowSpillProfiler *profiler,
    ShadowSpillBackendStream stream,
    const char *name
);

ShadowSpillProfilerRange shadowspill_profiler_range_begin(
    const ShadowSpillProfiler *profiler, const char *name
);

void shadowspill_profiler_range_end(
    const ShadowSpillProfiler *profiler, ShadowSpillProfilerRange range
);

#endif
