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

static ShadowSpillObject *find_unlocked(
    const ShadowSpillObjectTable *table,
    uint64_t object_id
) {
    if (table == NULL || table->by_id == NULL || table->bucket_count == 0U) {
        return NULL;
    }
    const uint64_t bucket = object_bucket(table, object_id);
    for (ShadowSpillObject *object = table->by_id[bucket]; object != NULL;
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
    ShadowSpillObject *object = table->owned_head;
    while (object != NULL) {
        ShadowSpillObject *next = object->ownership_next;
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

ShadowSpillObject *shadowspill_object_table_find(
    const ShadowSpillObjectTable *table,
    uint64_t object_id
) {
    return find_unlocked(table, object_id);
}

ShadowSpillObject *shadowspill_object_table_acquire(
    ShadowSpillObjectTable *table,
    uint64_t object_id
) {
    if (table == NULL) {
        return NULL;
    }
    pthread_rwlock_rdlock(&table->lock);
    ShadowSpillObject *object = find_unlocked(table, object_id);
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
    ShadowSpillObject *object
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
    ShadowSpillObject *object
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
    ShadowSpillObject **index_link = &table->by_id[bucket];
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

int shadowspill_object_table_rekey(
    ShadowSpillObjectTable *table,
    ShadowSpillObject *object,
    uint64_t replacement_object_id
) {
    if (table == NULL || object == NULL || table->by_id == NULL ||
        replacement_object_id == SHADOWSPILL_RUNTIME_NO_ID) {
        return -1;
    }
    pthread_rwlock_wrlock(&table->lock);
    if (object->ownership_previous_link == NULL ||
        find_unlocked(table, replacement_object_id) != NULL) {
        pthread_rwlock_unlock(&table->lock);
        return -1;
    }
    const uint64_t previous_bucket = object_bucket(table, object->object_id);
    ShadowSpillObject **previous_link = &table->by_id[previous_bucket];
    while (*previous_link != NULL && *previous_link != object) {
        previous_link = &(*previous_link)->id_index_next;
    }
    if (*previous_link != object) {
        pthread_rwlock_unlock(&table->lock);
        return -1;
    }
    *previous_link = object->id_index_next;
    object->object_id = replacement_object_id;
    const uint64_t replacement_bucket = object_bucket(table, replacement_object_id);
    object->id_index_next = table->by_id[replacement_bucket];
    table->by_id[replacement_bucket] = object;
    pthread_rwlock_unlock(&table->lock);
    return 0;
}

void shadowspill_object_retain(ShadowSpillObject *object) {
    if (object != NULL) {
        (void)atomic_fetch_add_explicit(
            &object->references, 1U, memory_order_relaxed
        );
    }
}

void shadowspill_object_release(ShadowSpillObject *object) {
    if (object == NULL || atomic_fetch_sub_explicit(
            &object->references, 1U, memory_order_acq_rel
        ) != 1U) {
        return;
    }
    free(object->locations);
    pthread_mutex_destroy(&object->lock);
    free(object);
}
