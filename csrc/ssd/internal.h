/* SSD storage implements the public pool and lane contracts. */
#ifndef SHADOWSPILL_SSD_INTERNAL_H
#define SHADOWSPILL_SSD_INTERNAL_H

#include <pthread.h>
#include <stdint.h>
#include <shadowspill/runtime/library.h>
#include <shadowspill/runtime/lane_base.h>
#include <shadowspill/ssd.h>

#define SHADOWSPILL_SSD_API __attribute__((visibility("default")))

typedef struct SSDRegion {
    int fd;
    /* The read-only image behind offsets below image_bytes, or -1. */
    int image_fd;
    void *base;
    uint64_t capacity;
    uint64_t file_bytes;
    uint64_t image_bytes;
    uint64_t alignment;
    ShadowSpillSSDConfiguration config;
    void *scratch;
    uint64_t scratch_bytes;
    uint64_t staging_reserved;
    pthread_mutex_t state_lock;
    pthread_mutex_t staging_lock;
    struct SSDRegion *next;
} SSDRegion;

extern const ShadowSpillPoolMemoryDescription shadowspill_ssd_pool_memory;
extern const ShadowSpillLaneDescription shadowspill_ssd_lanes[2];
/* A range's owner remains alive until its lanes have been destroyed. */
SSDRegion *ssd_region_find(void *base);
int ssd_staging_reserve(SSDRegion *region, uint64_t bytes);
void ssd_staging_release(SSDRegion *region, uint64_t bytes);
int ssd_direct_io(int fd, void *buffer, uint64_t bytes, uint64_t offset,
                  int writing);
/* The file and file offset behind a pool range: the image for ranges
   below image_bytes, which are not written, else the scratch file. A range
   crossing the image's end has no single file and fails. */
int ssd_locate(const SSDRegion *region, uint64_t offset, uint64_t bytes,
               int writing, int *fd, uint64_t *file_offset);

#endif
