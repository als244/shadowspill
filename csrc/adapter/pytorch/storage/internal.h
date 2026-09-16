#ifndef SHADOWSPILL_PYTORCH_STORAGE_INTERNAL_H
#define SHADOWSPILL_PYTORCH_STORAGE_INTERNAL_H

/*
 * PyTorch storages over runtime leases. objects.c holds the C primitives --
 * validate a CPU view against its lease, acquire objects for a stream, hand
 * one to the caller and take it back -- and cpu.cpp and device.cpp are the
 * torch operators over them, compiled when libtorch is found. Both halves
 * include this header, so it carries nothing C++ cannot parse.
 */

#include <shadowspill/pytorch_adapter.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The runtime this adapter is bound to. The storage operators are C++ and
   cannot parse the adapter struct, so they ask for it rather than read it. */
ShadowSpillRuntime *shadowspill_pytorch_bound_runtime(void);

#ifdef __cplusplus
}

/* Opens a profiler range for an operator's scope; a no-op without one. The
   range is the runtime's: the adapter keeps no profiler of its own. */
struct RangeGuard {
  explicit RangeGuard(const char* name)
      : runtime(shadowspill_pytorch_bound_runtime()),
        range(shadowspill_profiler_range_begin(runtime, name)) {}
  ~RangeGuard() { shadowspill_profiler_range_end(runtime, range); }

  ShadowSpillRuntime* runtime;

  ShadowSpillProfilerRange range;
};
#endif

#endif
