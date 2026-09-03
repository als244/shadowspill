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

void shadowspill_trace_append_stamped_enabled(
    ShadowSpillRuntime *runtime,
    ShadowSpillTraceEventKind kind,
    uint64_t task_id,
    uint64_t object_id,
    uint64_t allocation_id,
    uint64_t bytes,
    uint64_t detail_0,
    uint64_t detail_1,
    uint64_t stream_start_ns,
    uint64_t stream_end_ns
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



void shadowspill_profiler_name_current_thread(
    const ShadowSpillBackend *backend, const char *name
);

void shadowspill_profiler_name_stream(
    const ShadowSpillBackend *backend,
    ShadowSpillBackendStream stream,
    const char *name
);

ShadowSpillProfilerRange shadowspill_profiler_range_begin(
    const ShadowSpillBackend *backend, const char *name
);

void shadowspill_profiler_range_end(
    const ShadowSpillBackend *backend, ShadowSpillProfilerRange range
);

#endif
