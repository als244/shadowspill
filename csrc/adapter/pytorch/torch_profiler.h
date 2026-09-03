#ifndef SHADOWSPILL_PYTORCH_PROFILER_H
#define SHADOWSPILL_PYTORCH_PROFILER_H

#include <shadowspill/backend.h>

#ifdef __cplusplus
extern "C" {
#endif

ShadowSpillProfilerRange shadowspill_pytorch_profile_range_begin(
    const char *name
);
void shadowspill_pytorch_profile_range_end(ShadowSpillProfilerRange range);

#ifdef __cplusplus
}
#endif

#endif
