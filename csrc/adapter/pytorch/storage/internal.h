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
/* Opens a profiler range for an operator's scope; a no-op without one. */
struct RangeGuard {
  explicit RangeGuard(const char* name)
      : range(shadowspill_pytorch_profile_range_begin(name)) {}
  ~RangeGuard() { shadowspill_pytorch_profile_range_end(range); }

  ShadowSpillProfilerRange range;
};
#endif

#endif
