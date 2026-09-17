/* Which entry supplies a pool's memory, by kind. */
#include "../internal.h"

#include <stdlib.h>

/*
 * The runtime's own kinds are seeded first and a registered entry may not
 * displace one. A second claim on a kind is refused rather than shadowed:
 * order must never decide silently which entry a pool gets, because whichever
 * lost would be invisible and every allocation in that pool would quietly come
 * from the other.
 */

static int entry_is_valid(const ShadowSpillPoolMemoryDescription *entry) {
    return entry != NULL && entry->acquire != NULL && entry->release != NULL;
}

int shadowspill_pool_memory_table_initialize(
    ShadowSpillPoolMemoryTable *table,
    const ShadowSpillBackend *backend,
    const ShadowSpillPoolMemoryDescription *registered,
    uint32_t registered_count
) {
    if (table == NULL || backend == NULL ||
        (registered == NULL && registered_count != 0U)) {
        return -1;
    }
    const uint32_t builtin_count = 2U;
    table->entries = calloc(builtin_count + registered_count,
                            sizeof(*table->entries));
    if (table->entries == NULL) {
        return -1;
    }
    shadowspill_builtin_pool_memory_describe(backend, table->entries);
    table->count = builtin_count;
    for (uint32_t index = 0U; index < registered_count; ++index) {
        const ShadowSpillPoolMemoryDescription *entry = &registered[index];
        if (!entry_is_valid(entry) ||
            shadowspill_pool_memory_for_kind(table, entry->kind) != NULL) {
            shadowspill_pool_memory_table_destroy(table);
            return -1;
        }
        table->entries[table->count++] = *entry;
    }
    return 0;
}

void shadowspill_pool_memory_table_destroy(ShadowSpillPoolMemoryTable *table) {
    if (table == NULL) {
        return;
    }
    free(table->entries);
    table->entries = NULL;
    table->count = 0U;
}

const ShadowSpillPoolMemoryDescription *shadowspill_pool_memory_for_kind(
    const ShadowSpillPoolMemoryTable *table, uint8_t kind
) {
    if (table == NULL) {
        return NULL;
    }
    for (uint32_t index = 0U; index < table->count; ++index) {
        if (table->entries[index].kind == kind) {
            return &table->entries[index];
        }
    }
    return NULL;
}
