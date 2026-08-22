#ifndef SHADOWSPILL_PROFILER_H
#define SHADOWSPILL_PROFILER_H

#include <stdint.h>

#include <shadowspill/backend.h>

#ifdef __cplusplus
extern "C" {
#endif

/* A backend supplies this struct and the runtime checks it, so this is a
 * plugin contract like the ones in <shadowspill/backend.h> and versions on its
 * own rather than with the library. */
#define SHADOWSPILL_PROFILER_ABI_VERSION 2U

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
    void (*set_enabled)(void *context, uint8_t enabled);
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
