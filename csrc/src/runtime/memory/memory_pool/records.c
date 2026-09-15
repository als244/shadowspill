/* The lease and use records a pool owns, reserved once. */
#include "internal.h"

static void free_unowned_lease_records(ShadowSpillMemoryLease *records) {
    while (records != NULL) {
        ShadowSpillMemoryLease *next = records->free_record_next;
        free(records);
        records = next;
    }
}

static void free_unowned_use_records(ShadowSpillLeaseUseRecord *records) {
    while (records != NULL) {
        ShadowSpillLeaseUseRecord *next = records->free_next;
        free(records);
        records = next;
    }
}

ShadowSpillStatus shadowspill_memory_pool_reserve_lease_records(
    ShadowSpillMemoryPool *pool,
    uint64_t minimum_free_records
) {
    if (pool == NULL || !pool->initialized || minimum_free_records == 0U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&pool->lock);
    const uint64_t additional_leases = pool->lease_record_available <
            minimum_free_records
        ? minimum_free_records - pool->lease_record_available
        : 0U;
    const uint64_t additional_uses = pool->use_record_available <
            minimum_free_records
        ? minimum_free_records - pool->use_record_available
        : 0U;
    if (additional_leases > UINT64_MAX - pool->lease_record_capacity) {
        pthread_mutex_unlock(&pool->lock);
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    const uint64_t target_lease_capacity =
        pool->lease_record_capacity + additional_leases;
    if (target_lease_capacity > (UINT64_MAX - 2U) / 2U) {
        pthread_mutex_unlock(&pool->lock);
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    const uint64_t target_range_capacity =
        2U * target_lease_capacity + 2U;
    if (target_lease_capacity > SIZE_MAX / sizeof(ShadowSpillMemoryLease *) ||
        target_range_capacity > SIZE_MAX / sizeof(ShadowSpillRange)) {
        pthread_mutex_unlock(&pool->lock);
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    const int grow_frontier = pool->release_frontier_capacity <
        target_lease_capacity;
    const int grow_ranges = pool->release_range_capacity <
        target_range_capacity;
    if (additional_leases == 0U && additional_uses == 0U &&
        !grow_frontier && !grow_ranges) {
        pool->lease_records_sealed = 1U;
        pool->use_records_sealed = 1U;
        pthread_mutex_unlock(&pool->lock);
        return SHADOWSPILL_STATUS_OK;
    }
    pthread_mutex_unlock(&pool->lock);

    ShadowSpillMemoryLease *created_leases = NULL;
    uint64_t created_lease_count = 0U;
    while (created_lease_count < additional_leases) {
        ShadowSpillMemoryLease *record = calloc(1U, sizeof(*record));
        if (record == NULL) {
            free_unowned_lease_records(created_leases);
            return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        }
        record->free_record_next = created_leases;
        created_leases = record;
        ++created_lease_count;
    }
    ShadowSpillLeaseUseRecord *created_uses = NULL;
    uint64_t created_use_count = 0U;
    while (created_use_count < additional_uses) {
        ShadowSpillLeaseUseRecord *record = calloc(1U, sizeof(*record));
        if (record == NULL) {
            free_unowned_use_records(created_uses);
            free_unowned_lease_records(created_leases);
            return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        }
        record->free_next = created_uses;
        created_uses = record;
        ++created_use_count;
    }

    ShadowSpillMemoryLease **frontier = grow_frontier
        ? calloc((size_t)target_lease_capacity, sizeof(*frontier))
        : NULL;
    ShadowSpillRange *ranges = grow_ranges
        ? calloc((size_t)target_range_capacity, sizeof(*ranges))
        : NULL;
    if ((grow_frontier && frontier == NULL) ||
        (grow_ranges && ranges == NULL)) {
        free(ranges);
        free(frontier);
        free_unowned_use_records(created_uses);
        free_unowned_lease_records(created_leases);
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }

    pthread_mutex_lock(&pool->lock);
    ShadowSpillMemoryLease **old_frontier = NULL;
    ShadowSpillRange *old_ranges = NULL;
    while (created_leases != NULL) {
        ShadowSpillMemoryLease *next = created_leases->free_record_next;
        created_leases->metadata_owner = pool;
        atomic_init(&created_leases->references, 1U);
        created_leases->ownership_next = pool->owned_leases;
        pool->owned_leases = created_leases;
        created_leases->free_record_next = pool->free_lease_records;
        pool->free_lease_records = created_leases;
        created_leases = next;
    }
    while (created_uses != NULL) {
        ShadowSpillLeaseUseRecord *next = created_uses->free_next;
        created_uses->ownership_next = pool->owned_use_records;
        pool->owned_use_records = created_uses;
        created_uses->free_next = pool->free_use_records;
        pool->free_use_records = created_uses;
        created_uses = next;
    }
    pool->lease_record_capacity += created_lease_count;
    pool->lease_record_available += created_lease_count;
    pool->use_record_capacity += created_use_count;
    pool->use_record_available += created_use_count;
    if (frontier != NULL) {
        old_frontier = pool->release_frontier_workspace;
        pool->release_frontier_workspace = frontier;
        pool->release_frontier_capacity = target_lease_capacity;
    }
    if (ranges != NULL) {
        old_ranges = pool->release_range_workspace;
        pool->release_range_workspace = ranges;
        pool->release_range_capacity = target_range_capacity;
    }
    pool->lease_records_sealed = 1U;
    pool->use_records_sealed = 1U;
    pthread_mutex_unlock(&pool->lock);
    free(old_ranges);
    free(old_frontier);
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillLeaseUseRecord *shadowspill_memory_pool_acquire_use_record_locked(
    ShadowSpillMemoryPool *pool
) {
    if (pool == NULL || !pool->initialized) {
        return NULL;
    }
    ShadowSpillLeaseUseRecord *record = pool->free_use_records;
    if (record != NULL) {
        pool->free_use_records = record->free_next;
        record->free_next = NULL;
        --pool->use_record_available;
    } else if (pool->use_records_sealed) {
        ++pool->use_record_growth_rejections;
        return NULL;
    } else {
        record = calloc(1U, sizeof(*record));
        if (record == NULL) {
            return NULL;
        }
        record->ownership_next = pool->owned_use_records;
        pool->owned_use_records = record;
        ++pool->use_record_capacity;
    }
    ShadowSpillLeaseUseRecord *ownership_next = record->ownership_next;
    memset(record, 0, sizeof(*record));
    record->ownership_next = ownership_next;
    ++pool->use_record_in_use;
    if (pool->use_record_in_use > pool->use_record_peak_in_use) {
        pool->use_record_peak_in_use = pool->use_record_in_use;
    }
    return record;
}

int shadowspill_memory_pool_release_use_records_locked(
    ShadowSpillMemoryPool *pool,
    ShadowSpillLeaseUseRecord *records
) {
    if (pool == NULL) {
        return -1;
    }
    for (const ShadowSpillLeaseUseRecord *record = records;
         record != NULL; record = record->next) {
        if (record->event != NULL) {
            return -1;
        }
    }
    while (records != NULL) {
        ShadowSpillLeaseUseRecord *next = records->next;
        ShadowSpillLeaseUseRecord *ownership_next = records->ownership_next;
        memset(records, 0, sizeof(*records));
        records->ownership_next = ownership_next;
        records->free_next = pool->free_use_records;
        pool->free_use_records = records;
        ++pool->use_record_available;
        if (pool->use_record_in_use != 0U) {
            --pool->use_record_in_use;
        }
        records = next;
    }
    return 0;
}
