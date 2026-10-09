#define _GNU_SOURCE
#include "internal.h"

#include <errno.h>
#include <assert.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static pthread_mutex_t regions_lock = PTHREAD_MUTEX_INITIALIZER;
static SSDRegion *regions;

static uint64_t round_up(uint64_t bytes, uint64_t alignment) {
    return (bytes + alignment - 1U) / alignment * alignment;
}

static int temporary_file(const char *directory) {
    int fd = open(directory, O_TMPFILE | O_RDWR | O_DIRECT | O_CLOEXEC, 0600);
    if (fd >= 0) return fd;
    if (errno != EOPNOTSUPP && errno != EISDIR && errno != EINVAL) return -1;
    /* The fallback is unlinked before any allocation or transfer. There is
       no named pool to mistake for a checkpoint or leave after a crash. */
    const size_t length = strlen(directory) + sizeof("/.shadowspill-XXXXXX");
    char *path = malloc(length);
    if (path == NULL) return -1;
    (void)snprintf(path, length, "%s/.shadowspill-XXXXXX", directory);
    fd = mkostemp(path, O_DIRECT | O_CLOEXEC);
    if (fd >= 0 && unlink(path) != 0) {
        const int failure = errno;
        (void)close(fd);
        fd = -1;
        errno = failure;
    }
    free(path);
    return fd;
}

SSDRegion *ssd_region_find(void *base) {
    pthread_mutex_lock(&regions_lock);
    SSDRegion *region = regions;
    while (region != NULL && region->base != base) region = region->next;
    pthread_mutex_unlock(&regions_lock);
    return region;
}

int ssd_staging_reserve(SSDRegion *region, uint64_t bytes) {
    pthread_mutex_lock(&region->staging_lock);
    const int fits = bytes <= region->config.staging_bytes - region->staging_reserved;
    if (fits) region->staging_reserved += bytes;
    pthread_mutex_unlock(&region->staging_lock);
    if (!fits) {
        fprintf(stderr, "shadowspill SSD: staging budget exhausted requesting "
                "%" PRIu64 " bytes\n", bytes);
        return -1;
    }
    return 0;
}

void ssd_staging_release(SSDRegion *region, uint64_t bytes) {
    pthread_mutex_lock(&region->staging_lock);
    assert(bytes <= region->staging_reserved);
    region->staging_reserved -= bytes;
    pthread_mutex_unlock(&region->staging_lock);
}

/* A short direct I/O completion cannot be retried at an arbitrary byte
   offset without violating alignment. Treat it as a failed transfer. */
int ssd_direct_io(int fd, void *buffer, uint64_t bytes, uint64_t offset,
                  int writing) {
    ssize_t result;
    do {
        result = writing ? pwrite(fd, buffer, (size_t)bytes, (off_t)offset)
                         : pread(fd, buffer, (size_t)bytes, (off_t)offset);
    } while (result < 0 && errno == EINTR);
    if (result < 0 || (uint64_t)result != bytes) {
        fprintf(stderr, "shadowspill SSD: direct %s at %" PRIu64
                " for %" PRIu64 " bytes failed: %s\n",
                writing ? "write" : "read", offset, bytes,
                result < 0 ? strerror(errno) : "short completion");
        return -1;
    }
    return 0;
}

static int acquire(void *configuration, uint64_t capacity,
                   void **base, void **state) {
    if (base != NULL) *base = NULL;
    if (state != NULL) *state = NULL;
    const ShadowSpillSSDConfiguration *config = configuration;
    if (base == NULL || state == NULL || config == NULL ||
        config->directory == NULL || capacity == 0U || capacity > SIZE_MAX ||
        capacity > (uint64_t)INT64_MAX || config->chunk_bytes == 0U ||
        config->chunk_bytes > SIZE_MAX ||
        config->chunk_bytes > (uint64_t)SSIZE_MAX || config->queue_depth == 0U) {
        fprintf(stderr, "shadowspill SSD: invalid pool configuration\n");
        return -1;
    }
    SSDRegion *region = calloc(1U, sizeof(*region));
    if (region == NULL) return -1;
    region->fd = -1;
    region->config = *config;
    region->capacity = capacity;
    region->alignment = 4096U;
    if (pthread_mutex_init(&region->state_lock, NULL) != 0) {
        free(region);
        return -1;
    }
    if (pthread_mutex_init(&region->staging_lock, NULL) != 0) {
        pthread_mutex_destroy(&region->state_lock);
        free(region);
        return -1;
    }
    region->fd = temporary_file(config->directory);
    if (region->fd < 0) {
        fprintf(stderr, "shadowspill SSD: cannot create a direct-I/O temporary "
                "file in %s: %s\n", config->directory, strerror(errno));
        goto fail;
    }
#ifdef STATX_DIOALIGN
    struct statx details = {0};
    if (statx(region->fd, "", AT_EMPTY_PATH, STATX_DIOALIGN, &details) == 0 &&
        (details.stx_mask & STATX_DIOALIGN) != 0U) {
        if (details.stx_dio_mem_align == 0U || details.stx_dio_offset_align == 0U) {
            fprintf(stderr, "shadowspill SSD: filesystem does not support direct I/O\n");
            goto fail;
        }
        if (details.stx_dio_mem_align > region->alignment)
            region->alignment = details.stx_dio_mem_align;
        if (details.stx_dio_offset_align > region->alignment)
            region->alignment = details.stx_dio_offset_align;
    }
#endif
    if (capacity > (uint64_t)INT64_MAX - (region->alignment - 1U) ||
        (region->alignment & (region->alignment - 1U)) != 0U ||
        config->chunk_bytes % region->alignment != 0U ||
        config->chunk_bytes < region->alignment) {
        fprintf(stderr, "shadowspill SSD: chunk size must satisfy direct-I/O alignment\n");
        goto fail;
    }
    region->scratch_bytes = config->chunk_bytes;
    if (region->scratch_bytes > config->staging_bytes ||
        posix_memalign(&region->scratch, (size_t)region->alignment,
                       (size_t)region->scratch_bytes) != 0) {
        fprintf(stderr, "shadowspill SSD: cannot reserve bounded import staging\n");
        goto fail;
    }
    region->staging_reserved = region->scratch_bytes;
    region->file_bytes = round_up(capacity, region->alignment);
    const int allocated = posix_fallocate(region->fd, 0, (off_t)region->file_bytes);
    if (allocated != 0) {
        fprintf(stderr, "shadowspill SSD: cannot reserve %" PRIu64
                " bytes in %s: %s\n", region->file_bytes, config->directory,
                strerror(allocated));
        goto fail;
    }
    /* Only an address-space token: the allocator does offset arithmetic,
       while read/write and lanes access the file. No host payload mapping. */
    region->base = mmap(NULL, (size_t)capacity, PROT_NONE,
                         MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (region->base == MAP_FAILED) {
        region->base = NULL;
        goto fail;
    }
    pthread_mutex_lock(&regions_lock);
    region->next = regions;
    regions = region;
    pthread_mutex_unlock(&regions_lock);
    *base = region->base;
    *state = region;
    return 0;
fail:
    if (region->fd >= 0) (void)close(region->fd);
    free(region->scratch);
    pthread_mutex_destroy(&region->staging_lock);
    pthread_mutex_destroy(&region->state_lock);
    free(region);
    return -1;
}

static int release(void *state, void *base, uint64_t capacity) {
    SSDRegion *region = state;
    if (region == NULL || region->base != base || region->capacity != capacity)
        return -1;
    pthread_mutex_lock(&regions_lock);
    SSDRegion **link = &regions;
    while (*link != NULL && *link != region) link = &(*link)->next;
    if (*link == region) *link = region->next;
    pthread_mutex_unlock(&regions_lock);
    const int unmapped = munmap(base, (size_t)capacity);
    const int closed = close(region->fd);
    free(region->scratch);
    pthread_mutex_destroy(&region->staging_lock);
    pthread_mutex_destroy(&region->state_lock);
    free(region);
    return unmapped == 0 && closed == 0 ? 0 : -1;
}

static int state_io(SSDRegion *region, uint64_t offset, void *buffer,
                    uint64_t bytes, int writing) {
    if (region == NULL || offset > region->capacity ||
        bytes > region->capacity - offset || (bytes != 0U && buffer == NULL))
        return -1;
    pthread_mutex_lock(&region->state_lock);
    int status = 0;
    uint64_t moved = 0U;
    while (moved < bytes && status == 0) {
        const uint64_t at = offset + moved;
        const uint64_t prefix = at % region->alignment;
        const uint64_t available = region->scratch_bytes - prefix;
        const uint64_t payload = bytes - moved < available ? bytes - moved : available;
        const uint64_t transfer = round_up(prefix + payload, region->alignment);
        const uint64_t start = at - prefix;
        if (!writing || prefix != 0U || transfer != payload)
            status = ssd_direct_io(region->fd, region->scratch, transfer, start, 0);
        if (status != 0) break;
        if (writing) {
            memcpy((char *)region->scratch + prefix, (char *)buffer + moved,
                   (size_t)payload);
            status = ssd_direct_io(region->fd, region->scratch, transfer, start, 1);
        } else {
            memcpy((char *)buffer + moved, (char *)region->scratch + prefix,
                   (size_t)payload);
        }
        moved += payload;
    }
    pthread_mutex_unlock(&region->state_lock);
    return status;
}

static int write_state(void *state, uint64_t offset, const void *source,
                       uint64_t bytes) {
    return state_io(state, offset, (void *)(uintptr_t)source, bytes, 1);
}

static int read_state(void *state, uint64_t offset, void *destination,
                      uint64_t bytes) {
    return state_io(state, offset, destination, bytes, 0);
}

const ShadowSpillPoolMemoryDescription shadowspill_ssd_pool_memory = {
    .kind = SHADOWSPILL_SSD_POOL_KIND,
    .acquire = acquire,
    .release = release,
    .write = write_state,
    .read = read_state,
};
