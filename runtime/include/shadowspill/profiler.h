#ifndef SHADOWSPILL_PROFILER_H
#define SHADOWSPILL_PROFILER_H

#include <stdint.h>

#include <shadowspill/backend.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SHADOWSPILL_PROFILER_ABI_VERSION 1U

typedef uint64_t ShadowSpillProfilerRange;

/*
 * Optional observability callbacks supplied by a concrete backend. The
 * neutral runtime treats names and ranges as best-effort diagnostics: a
 * missing callback is a no-op and never changes execution semantics.
 */
typedef struct ShadowSpillProfiler {
    uint32_t abi_version;
    void *context;
    void (*name_current_thread)(void *context, const char *name);
    void (*name_stream)(
        void *context,
        ShadowSpillBackendStream stream,
        const char *name
    );
    ShadowSpillProfilerRange (*range_begin)(
        void *context,
        const char *name
    );
    void (*range_end)(void *context, ShadowSpillProfilerRange range);
} ShadowSpillProfiler;

#ifdef __cplusplus
}
#endif

#endif
