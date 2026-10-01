/* Startup-only host placement. No accelerator or framework dependencies. */
#ifndef SHADOWSPILL_NUMA_H
#define SHADOWSPILL_NUMA_H
#include <shadowspill/backend.h>

/* Returns the preferred host node, or -1 when disabled/unavailable. */
int shadowspill_numa_initialize(const ShadowSpillBackend *backend, int disabled);
void shadowspill_numa_place(void *address, uint64_t bytes, int node);
void shadowspill_numa_verify(void *address, uint64_t bytes, int node);
#endif
