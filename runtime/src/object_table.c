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

int shadowspill_object_table_initialize(
    ShadowSpillObjectTable *table,
    uint64_t bucket_count
) {
    if (table == NULL || bucket_count == 0U || bucket_count > SIZE_MAX) {
        return -1;
    }
    table->by_id = calloc((size_t)bucket_count, sizeof(*table->by_id));
    if (table->by_id == NULL) {
        return -1;
    }
    table->bucket_count = bucket_count;
    return 0;
}

void shadowspill_object_table_destroy(ShadowSpillObjectTable *table) {
    if (table == NULL) {
        return;
    }
    ShadowSpillObjectRecord *object = table->owned_head;
    while (object != NULL) {
        ShadowSpillObjectRecord *next = object->ownership_next;
        free(object);
        object = next;
    }
    free(table->by_id);
    *table = (ShadowSpillObjectTable){0};
}

ShadowSpillObjectRecord *shadowspill_object_table_find(
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

int shadowspill_object_table_insert(
    ShadowSpillObjectTable *table,
    ShadowSpillObjectRecord *object
) {
    if (table == NULL || object == NULL || table->by_id == NULL ||
        shadowspill_object_table_find(table, object->object_id) != NULL) {
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
    const uint64_t bucket = object_bucket(table, object->object_id);
    ShadowSpillObjectRecord **index_link = &table->by_id[bucket];
    while (*index_link != NULL && *index_link != object) {
        index_link = &(*index_link)->id_index_next;
    }
    if (*index_link != object) {
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
    return 0;
}
