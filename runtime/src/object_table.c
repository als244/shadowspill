#include "internal.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

static uint64_t object_bucket(
    const ShadowSpillObjectTable *table,
    uint64_t object_id
) {
    object_id ^= object_id >> 33U;
    object_id *= UINT64_C(0xff51afd7ed558ccd);
    object_id ^= object_id >> 33U;
    return object_id % table->bucket_count;
}

static ShadowSpillObjectRecord *find_unlocked(
    const ShadowSpillObjectTable *table,
    uint64_t object_id
) {
    if (table == NULL || table->by_id == NULL || table->bucket_count == 0U) {
        return NULL;
    }
    const uint64_t bucket = object_bucket(table, object_id);
    for (ShadowSpillObjectRecord *object = table->by_id[bucket]; object != NULL;
         object = object->id_index_next) {
        if (object->object_id == object_id) {
            return object;
        }
    }
    return NULL;
}

int shadowspill_object_table_initialize(
    ShadowSpillObjectTable *table,
    uint64_t bucket_count
) {
    if (table == NULL || bucket_count == 0U || bucket_count > SIZE_MAX) {
        return -1;
    }
    if (pthread_rwlock_init(&table->lock, NULL) != 0) {
        return -1;
    }
    table->lock_initialized = 1U;
    table->by_id = calloc((size_t)bucket_count, sizeof(*table->by_id));
    if (table->by_id == NULL) {
        pthread_rwlock_destroy(&table->lock);
        table->lock_initialized = 0U;
        return -1;
    }
    table->bucket_count = bucket_count;
    return 0;
}

void shadowspill_object_table_destroy(ShadowSpillObjectTable *table) {
    if (table == NULL) {
        return;
    }
    if (!table->lock_initialized) {
        free(table->by_id);
        *table = (ShadowSpillObjectTable){0};
        return;
    }
    ShadowSpillObjectRecord *object = table->owned_head;
    while (object != NULL) {
        ShadowSpillObjectRecord *next = object->ownership_next;
        object->ownership_next = NULL;
        object->ownership_previous_link = NULL;
        object->id_index_next = NULL;
        atomic_store_explicit(&object->detached, 1U, memory_order_release);
        shadowspill_object_release(object);
        object = next;
    }
    free(table->by_id);
    pthread_rwlock_destroy(&table->lock);
    *table = (ShadowSpillObjectTable){0};
}

ShadowSpillObjectRecord *shadowspill_object_table_find(
    const ShadowSpillObjectTable *table,
    uint64_t object_id
) {
    return find_unlocked(table, object_id);
}

ShadowSpillObjectRecord *shadowspill_object_table_acquire(
    ShadowSpillObjectTable *table,
    uint64_t object_id
) {
    if (table == NULL) {
        return NULL;
    }
    pthread_rwlock_rdlock(&table->lock);
    ShadowSpillObjectRecord *object = find_unlocked(table, object_id);
    if (object != NULL && atomic_load_explicit(
            &object->detached, memory_order_acquire
        ) == 0U) {
        shadowspill_object_retain(object);
    } else {
        object = NULL;
    }
    pthread_rwlock_unlock(&table->lock);
    return object;
}

int shadowspill_object_table_insert(
    ShadowSpillObjectTable *table,
    ShadowSpillObjectRecord *object
) {
    if (table == NULL || object == NULL || table->by_id == NULL) {
        return -1;
    }
    pthread_rwlock_wrlock(&table->lock);
    if (find_unlocked(table, object->object_id) != NULL) {
        pthread_rwlock_unlock(&table->lock);
        return -1;
    }
    const uint64_t bucket = object_bucket(table, object->object_id);
    object->id_index_next = table->by_id[bucket];
    table->by_id[bucket] = object;

    object->ownership_next = table->owned_head;
    object->ownership_previous_link = &table->owned_head;
    if (object->ownership_next != NULL) {
        object->ownership_next->ownership_previous_link = &object->ownership_next;
    }
    table->owned_head = object;
    pthread_rwlock_unlock(&table->lock);
    return 0;
}

int shadowspill_object_table_remove(
    ShadowSpillObjectTable *table,
    ShadowSpillObjectRecord *object
) {
    if (table == NULL || object == NULL || table->by_id == NULL ||
        object->ownership_previous_link == NULL) {
        return -1;
    }
    pthread_rwlock_wrlock(&table->lock);
    if (object->ownership_previous_link == NULL) {
        pthread_rwlock_unlock(&table->lock);
        return -1;
    }
    const uint64_t bucket = object_bucket(table, object->object_id);
    ShadowSpillObjectRecord **index_link = &table->by_id[bucket];
    while (*index_link != NULL && *index_link != object) {
        index_link = &(*index_link)->id_index_next;
    }
    if (*index_link != object) {
        pthread_rwlock_unlock(&table->lock);
        return -1;
    }
    *index_link = object->id_index_next;
    object->id_index_next = NULL;

    *object->ownership_previous_link = object->ownership_next;
    if (object->ownership_next != NULL) {
        object->ownership_next->ownership_previous_link =
            object->ownership_previous_link;
    }
    object->ownership_next = NULL;
    object->ownership_previous_link = NULL;
    atomic_store_explicit(&object->detached, 1U, memory_order_release);
    pthread_rwlock_unlock(&table->lock);
    return 0;
}

void shadowspill_object_retain(ShadowSpillObjectRecord *object) {
    if (object != NULL) {
        (void)atomic_fetch_add_explicit(
            &object->references, 1U, memory_order_relaxed
        );
    }
}

void shadowspill_object_release(ShadowSpillObjectRecord *object) {
    if (object == NULL || atomic_fetch_sub_explicit(
            &object->references, 1U, memory_order_acq_rel
        ) != 1U) {
        return;
    }
    pthread_cond_destroy(&object->state_changed);
    pthread_mutex_destroy(&object->lock);
    free(object);
}
