/* Configuration for the Linux SSD pool and its fetch/evict lanes. */
#ifndef SHADOWSPILL_SSD_H
#define SHADOWSPILL_SSD_H

#include <stdint.h>

#define SHADOWSPILL_SSD_POOL_KIND 3U

typedef struct ShadowSpillSSDConfiguration {
    /* Existing directory on a filesystem supporting direct I/O/preallocation. */
    const char *directory;
    /* Host payload cap shared by state I/O and every lane using this pool. */
    uint64_t staging_bytes;
    /* Positive, divisible by the filesystem's direct-I/O alignment (>= 4096). */
    uint64_t chunk_bytes;
    /* Positive number of staging slots per directional lane. */
    uint32_t queue_depth;
} ShadowSpillSSDConfiguration;

#endif
