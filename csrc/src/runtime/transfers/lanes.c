/* Which lane serves a directional pool-kind pair. */
#include "../internal.h"

#include <stdlib.h>

/*
 * The built-ins are seeded first and a registered entry may not displace one,
 * so the table is append-only and the first claim on a pair wins -- but a
 * second claim is refused rather than shadowed. Order must never decide
 * silently which lane a route gets: a config that registers a pair the runtime
 * already serves is a mistake worth a create-time failure, not a surprise at
 * the first transfer.
 */

static int claims_pair(
    const ShadowSpillLaneDescription *entry, uint8_t from_kind, uint8_t to_kind
) {
    return entry->from_kind == from_kind && entry->to_kind == to_kind;
}

static int entry_is_valid(const ShadowSpillLaneDescription *entry) {
    if (entry == NULL || entry->operations == NULL || entry->create == NULL) {
        return 0;
    }
    const ShadowSpillLaneOperations *operations = entry->operations;
    /* `transfer` and `timing` are what a lane may leave out; the rest is what
       the runtime cannot proceed without. */
    return operations->wait != NULL && operations->copy != NULL &&
           operations->signal != NULL && operations->synchronize != NULL &&
           operations->destroy != NULL;
}

int shadowspill_lane_table_initialize(
    ShadowSpillLaneTable *table,
    ShadowSpillRuntime *runtime,
    const ShadowSpillLaneDescription *registered,
    uint32_t registered_count
) {
    if (table == NULL || runtime == NULL ||
        (registered == NULL && registered_count != 0U)) {
        return -1;
    }
    const uint32_t builtin_count = 2U;
    const uint32_t total = builtin_count + registered_count;
    table->entries = calloc(total, sizeof(*table->entries));
    if (table->entries == NULL) {
        return -1;
    }
    shadowspill_pinned_host_device_lanes_describe(runtime, table->entries);
    table->count = builtin_count;
    for (uint32_t index = 0U; index < registered_count; ++index) {
        const ShadowSpillLaneDescription *entry = &registered[index];
        if (!entry_is_valid(entry)) {
            shadowspill_lane_table_destroy(table);
            return -1;
        }
        if (shadowspill_lane_for_kinds(
                table, entry->from_kind, entry->to_kind
            ) != NULL) {
            /* Two lanes for one pair. Refused here rather than resolved by
               position, because whichever won would be invisible. */
            shadowspill_lane_table_destroy(table);
            return -1;
        }
        table->entries[table->count++] = *entry;
    }
    return 0;
}

void shadowspill_lane_table_destroy(ShadowSpillLaneTable *table) {
    if (table == NULL) {
        return;
    }
    free(table->entries);
    table->entries = NULL;
    table->count = 0U;
}

const ShadowSpillLaneDescription *shadowspill_lane_for_kinds(
    const ShadowSpillLaneTable *table, uint8_t from_kind, uint8_t to_kind
) {
    if (table == NULL) {
        return NULL;
    }
    for (uint32_t index = 0U; index < table->count; ++index) {
        if (claims_pair(&table->entries[index], from_kind, to_kind)) {
            return &table->entries[index];
        }
    }
    return NULL;
}
