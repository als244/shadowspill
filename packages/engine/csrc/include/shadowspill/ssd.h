/* Configuration for the Linux SSD pool and its fetch/evict lanes. */
#ifndef SHADOWSPILL_SSD_H
#define SHADOWSPILL_SSD_H

#include <stdint.h>

#define SHADOWSPILL_SSD_POOL_KIND 3U

/* How a pool uses its image file, if it has one. */
#define SHADOWSPILL_SSD_IMAGE_NONE 0U
#define SHADOWSPILL_SSD_IMAGE_CREATE 1U
#define SHADOWSPILL_SSD_IMAGE_ADOPT 2U

typedef struct ShadowSpillSSDConfiguration {
    /* Existing directory on a filesystem supporting direct I/O/preallocation. */
    const char *directory;
    /* Host payload cap shared by state I/O and every lane using this pool. */
    uint64_t staging_bytes;
    /* Positive, divisible by the filesystem's direct-I/O alignment (>= 4096). */
    uint64_t chunk_bytes;
    /* Positive number of staging slots per directional lane. */
    uint32_t queue_depth;
    /* NONE: the pool is one unlinked temporary file in directory. CREATE: the
       pool is the named file image_path, created here and kept at release,
       so state imported into it outlives the process. ADOPT: the existing
       image is opened read-only as the pool's first image-size bytes and the
       rest of the capacity is an unlinked scratch file in directory; a write
       into the image fails. */
    uint32_t image_mode;
    /* The image file for CREATE or ADOPT; ignored for NONE. */
    const char *image_path;
} ShadowSpillSSDConfiguration;

#endif
