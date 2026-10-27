/* Bounded import/export of idle spill payloads, independent of pool kind. */
#include "../internal.h"

#include <stdint.h>
#include <string.h>

/* Where a lease sits in its pool.
   Both directions across the pool's edge need this and neither may use the
   lease pointer instead: for a kind whose region lives on another machine the
   pointer is meaningful only to that kind, while the offset is meaningful to
   everyone. */
static uint64_t lease_offset(
    const ShadowSpillMemoryPool *pool,
    const ShadowSpillObjectLocation *location
) {
    return (uint64_t)(
        (const char *)location->lease->pointer - (const char *)pool->base
    );
}

static ShadowSpillStatus write_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    uint32_t pool_id,
    const void *source,
    uint64_t bytes,
    uint64_t offset,
    int exact
) {
    if (shadowspill_runtime_pool(runtime, pool_id) == NULL || bytes > SIZE_MAX ||
        (bytes != 0U && source == NULL)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillStatus status = shadowspill_current_status_locked(runtime);
    ShadowSpillObject *object = shadowspill_find_object(runtime, object_id);
    if (status != SHADOWSPILL_STATUS_OK) {
        goto done;
    }
    const uint64_t size = object == NULL ? 0U : shadowspill_object_pool_bytes(
        object, shadowspill_runtime_pool(runtime, pool_id)
    );
    if (object == NULL || offset > size || bytes > size - offset ||
        (exact && bytes != size) ||
        shadowspill_object_location(object, pool_id)->lease == NULL ||
        object->residency != SHADOWSPILL_OBJECT_SPILL_ONLY ||
        object->allocation_id != SHADOWSPILL_RUNTIME_NO_ID) {
        status = SHADOWSPILL_STATUS_INVALID_STATE;
        goto done;
    }
    if (bytes != 0U) {
        ShadowSpillObjectLocation *location = shadowspill_object_location(
            object, pool_id
        );
        ShadowSpillMemoryPool *const pool =
            shadowspill_runtime_pool(runtime, pool_id);
        if (pool->memory.write != NULL) {
            /* A kind whose region this process cannot address writes it
               itself; everyone else is an ordinary copy. */
            if (pool->memory.write(
                    pool->memory_state,
                    lease_offset(pool, location) + offset,
                    source,
                    bytes
                ) != 0) {
                status = SHADOWSPILL_STATUS_BACKEND_FAILURE;
                goto done;
            }
        } else {
            memcpy((char *)location->lease->pointer + offset, source, (size_t)bytes);
        }
    }
    ShadowSpillObjectLocation *location = shadowspill_object_location(
        object, pool_id
    );
    location->current = 1U;
    location->version = object->authoritative_version;

done:
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

static ShadowSpillStatus read_object(
    ShadowSpillRuntime *runtime,
    uint64_t object_id,
    uint32_t pool_id,
    void *destination,
    uint64_t bytes,
    uint64_t offset,
    int exact
) {
    if (shadowspill_runtime_pool(runtime, pool_id) == NULL || bytes > SIZE_MAX ||
        (bytes != 0U && destination == NULL)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&runtime->mutex);
    ShadowSpillStatus status = shadowspill_current_status_locked(runtime);
    ShadowSpillObject *object = shadowspill_find_object(runtime, object_id);
    ShadowSpillObjectLocation *location = object == NULL
        ? NULL : shadowspill_object_location(object, pool_id);
    if (status != SHADOWSPILL_STATUS_OK) {
        goto read_done;
    }
    /* The requirement is that the copy being read is the authoritative one: it
       exists, it is current, and its version is the object's. Whether a copy
       also lives somewhere else does not bear on that. Requiring sole residency
       as well conflated "this copy is authoritative" with "this is the only
       copy", so a caller could not read state that was also on the device --
       which a checkpoint has to be able to do without first giving up residency
       to get at it. */
    const uint64_t size = object == NULL ? 0U : shadowspill_object_pool_bytes(
        object, shadowspill_runtime_pool(runtime, pool_id)
    );
    if (object == NULL || offset > size || bytes > size - offset ||
        (exact && bytes != size) ||
        location->lease == NULL || !location->current ||
        location->version != object->authoritative_version) {
        status = SHADOWSPILL_STATUS_INVALID_STATE;
        goto read_done;
    }
    if (bytes != 0U) {
        ShadowSpillMemoryPool *const pool =
            shadowspill_runtime_pool(runtime, pool_id);
        if (pool->memory.read != NULL) {
            if (pool->memory.read(
                    pool->memory_state,
                    lease_offset(pool, location) + offset,
                    destination,
                    bytes
                ) != 0) {
                status = SHADOWSPILL_STATUS_BACKEND_FAILURE;
                goto read_done;
            }
        } else {
            memcpy(destination, (const char *)location->lease->pointer + offset,
                   (size_t)bytes);
        }
    }

read_done:
    pthread_mutex_unlock(&runtime->mutex);
    return status;
}

ShadowSpillStatus shadowspill_write_object(
    ShadowSpillRuntime *runtime, uint64_t object_id, uint32_t pool_id,
    const void *source, uint64_t bytes
) {
    return write_object(runtime, object_id, pool_id, source, bytes, 0U, 1);
}

ShadowSpillStatus shadowspill_read_object(
    ShadowSpillRuntime *runtime, uint64_t object_id, uint32_t pool_id,
    void *destination, uint64_t bytes
) {
    return read_object(runtime, object_id, pool_id, destination, bytes, 0U, 1);
}

ShadowSpillStatus shadowspill_write_object_range(
    ShadowSpillRuntime *runtime, uint64_t object_id, uint32_t pool_id,
    const void *source, uint64_t offset, uint64_t bytes
) {
    return write_object(runtime, object_id, pool_id, source, bytes, offset, 0);
}

ShadowSpillStatus shadowspill_read_object_range(
    ShadowSpillRuntime *runtime, uint64_t object_id, uint32_t pool_id,
    void *destination, uint64_t offset, uint64_t bytes
) {
    return read_object(runtime, object_id, pool_id, destination, bytes, offset, 0);
}
