#define _POSIX_C_SOURCE 200809L

#include "portfolio_internal.h"
#include "residency_internal.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct SimulationWorkspace {
    ShadowSpillTaskInterval *tasks;
    ShadowSpillTransferInterval *transfers;
    ShadowSpillDevicePeak *peaks;
    uint32_t task_capacity;
    uint32_t transfer_capacity;
    uint32_t device_capacity;
} SimulationWorkspace;

typedef struct HashSlot {
    uint64_t hash;
    uint32_t entry_plus_one;
} HashSlot;

typedef struct HashIndex {
    HashSlot *slots;
    uint32_t capacity;
    uint32_t count;
} HashIndex;

typedef struct ResidencyCacheEntry {
    uint64_t hash;
    uint8_t strategy;
    uint64_t *extra_pressure;
    uint8_t *resident;
    uint8_t *breaks;
    ShadowSpillPlannerStatus status;
    uint32_t error_device;
    int32_t error_boundary;
    uint64_t required_bytes;
    uint64_t capacity_bytes;
} ResidencyCacheEntry;

typedef struct ResidencyCache {
    ResidencyCacheEntry *entries;
    uint32_t count;
    uint32_t capacity;
    size_t cell_count;
    size_t pressure_count;
    HashIndex index;
} ResidencyCache;

typedef struct ScheduleCacheEntry {
    uint64_t hash;
    uint8_t rule;
    uint8_t coalesced;
    uint8_t prefetch_headroom;
    uint8_t *resident;
    uint8_t *breaks;
    ShadowSpillDenseSchedule schedule;
} ScheduleCacheEntry;

typedef struct ScheduleCache {
    ScheduleCacheEntry *entries;
    uint32_t count;
    uint32_t capacity;
    size_t cell_count;
    HashIndex index;
} ScheduleCache;

typedef struct SimulationCacheEntry {
    uint64_t hash;
    ShadowSpillDenseSchedule schedule;
    ShadowSpillSimulationResult result;
} SimulationCacheEntry;

typedef struct SimulationCache {
    SimulationCacheEntry *entries;
    uint32_t count;
    uint32_t capacity;
    HashIndex index;
} SimulationCache;

typedef struct CandidateWorkspace {
    uint8_t *resident;
    uint8_t *breaks;
    uint8_t *base_resident;
    uint8_t *base_breaks;
    uint64_t *extra_pressure;
    ShadowSpillScheduleStorage schedule;
    ShadowSpillScheduleStorage selected;
    SimulationWorkspace simulation;
    ResidencyCache residency_cache;
    ScheduleCache schedule_cache;
    SimulationCache simulation_cache;
    ShadowSpillResidencyWorkspace *residency_workspace;
    uint64_t residency_cache_hits;
    uint64_t residency_cache_misses;
    uint64_t schedule_emissions;
    uint64_t schedule_cache_hits;
    uint64_t simulation_calls;
    uint64_t simulation_cache_hits;
    uint64_t residency_time_ns;
    uint64_t schedule_time_ns;
    uint64_t simulation_time_ns;
    uint64_t digest_time_ns;
} CandidateWorkspace;

static uint64_t monotonic_time_ns(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) {
        return 0U;
    }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) +
        (uint64_t)value.tv_nsec;
}

static uint64_t hash_bytes(uint64_t hash, const void *data, size_t size) {
    const uint8_t *bytes = data;
    for (size_t index = 0U; index < size; ++index) {
        hash ^= bytes[index];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static uint64_t hash_value(uint64_t hash, uint64_t value) {
    return hash_bytes(hash, &value, sizeof(value));
}

static uint32_t hash_slot(uint64_t hash, uint32_t capacity) {
    hash ^= hash >> 33U;
    hash *= UINT64_C(0xff51afd7ed558ccd);
    hash ^= hash >> 33U;
    return (uint32_t)hash & (capacity - 1U);
}

static int hash_index_resize(HashIndex *index, uint32_t capacity) {
    HashSlot *slots = calloc(capacity, sizeof(*slots));
    if (slots == NULL) {
        return -1;
    }
    for (uint32_t old = 0U; old < index->capacity; ++old) {
        HashSlot value = index->slots[old];
        if (value.entry_plus_one == 0U) {
            continue;
        }
        uint32_t slot = hash_slot(value.hash, capacity);
        while (slots[slot].entry_plus_one != 0U) {
            slot = (slot + 1U) & (capacity - 1U);
        }
        slots[slot] = value;
    }
    free(index->slots);
    index->slots = slots;
    index->capacity = capacity;
    return 0;
}

static int hash_index_insert(
    HashIndex *index,
    uint64_t hash,
    uint32_t entry_index
) {
    if (index->capacity == 0U ||
        (uint64_t)(index->count + 1U) * 10U >=
            (uint64_t)index->capacity * 7U) {
        uint32_t capacity = index->capacity == 0U ? 64U : index->capacity * 2U;
        if (capacity < index->capacity || hash_index_resize(index, capacity) != 0) {
            return -1;
        }
    }
    uint32_t slot = hash_slot(hash, index->capacity);
    while (index->slots[slot].entry_plus_one != 0U) {
        slot = (slot + 1U) & (index->capacity - 1U);
    }
    index->slots[slot] = (HashSlot){
        .hash = hash,
        .entry_plus_one = entry_index + 1U,
    };
    ++index->count;
    return 0;
}

static uint32_t hash_index_start(const HashIndex *index, uint64_t hash) {
    return index->capacity == 0U
        ? UINT32_MAX
        : hash_slot(hash, index->capacity);
}

static uint32_t hash_index_next(const HashIndex *index, uint32_t slot) {
    return (slot + 1U) & (index->capacity - 1U);
}

static uint64_t residency_cache_hash(
    const CandidateWorkspace *workspace,
    uint8_t strategy
) {
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = hash_bytes(hash, &strategy, sizeof(strategy));
    return hash_bytes(
        hash,
        workspace->extra_pressure,
        workspace->residency_cache.pressure_count *
            sizeof(*workspace->extra_pressure)
    );
}

static uint64_t schedule_cache_hash(
    const CandidateWorkspace *workspace,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint8_t rule,
    uint8_t coalesced,
    uint8_t prefetch_headroom
) {
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = hash_bytes(hash, &rule, sizeof(rule));
    hash = hash_bytes(hash, &coalesced, sizeof(coalesced));
    hash = hash_bytes(hash, &prefetch_headroom, sizeof(prefetch_headroom));
    hash = hash_bytes(hash, resident, workspace->schedule_cache.cell_count);
    return hash_bytes(hash, breaks, workspace->schedule_cache.cell_count);
}

static uint64_t dense_schedule_hash(const ShadowSpillDenseSchedule *schedule) {
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = hash_value(hash, schedule->action_count);
    hash = hash_bytes(
        hash,
        schedule->action_trigger_tasks,
        (size_t)schedule->action_count * sizeof(*schedule->action_trigger_tasks)
    );
    hash = hash_bytes(
        hash,
        schedule->action_aliases,
        (size_t)schedule->action_count * sizeof(*schedule->action_aliases)
    );
    hash = hash_bytes(
        hash,
        schedule->action_kinds,
        (size_t)schedule->action_count * sizeof(*schedule->action_kinds)
    );
    hash = hash_value(hash, schedule->initial_count);
    hash = hash_bytes(
        hash,
        schedule->initial_aliases,
        (size_t)schedule->initial_count * sizeof(*schedule->initial_aliases)
    );
    hash = hash_bytes(
        hash,
        schedule->initial_locations,
        (size_t)schedule->initial_count * sizeof(*schedule->initial_locations)
    );
    hash = hash_value(hash, schedule->final_count);
    hash = hash_bytes(
        hash,
        schedule->final_aliases,
        (size_t)schedule->final_count * sizeof(*schedule->final_aliases)
    );
    return hash_bytes(
        hash,
        schedule->final_locations,
        (size_t)schedule->final_count * sizeof(*schedule->final_locations)
    );
}

static int multiply_u32(uint32_t left, uint32_t right, uint32_t *result) {
    uint64_t value = (uint64_t)left * right;
    if (value > UINT32_MAX) {
        return -1;
    }
    *result = (uint32_t)value;
    return 0;
}

static int strategy_valid(uint8_t strategy) {
    return strategy <= SHADOWSPILL_RESIDENCY_RELAXED_STALL;
}

static int rule_valid(uint8_t rule) {
    return rule <= SHADOWSPILL_PREFETCH_LATEST_SAFE;
}

static int context_valid(
    const ShadowSpillPressureFitContext *context,
    const ShadowSpillPressureFitContextOptions *options
) {
    if (context == NULL || options == NULL || context->residency == NULL ||
        context->simulation == NULL || context->seed_resident == NULL ||
        context->seed_breaks == NULL || context->alias_json_names == NULL ||
        context->task_json_names == NULL ||
        context->abi_version != SHADOWSPILL_PLANNER_ABI_VERSION ||
        context->residency->abi_version != SHADOWSPILL_PLANNER_ABI_VERSION ||
        context->simulation->abi_version != SHADOWSPILL_SIMULATOR_ABI_VERSION ||
        context->simulation->task_count == 0U ||
        options->residency_strategies == NULL ||
        options->residency_strategy_count == 0U ||
        options->prefetch_rules == NULL || options->prefetch_rule_count == 0U) {
        return 0;
    }
    for (uint32_t index = 0U; index < options->residency_strategy_count;
         ++index) {
        if (!strategy_valid(options->residency_strategies[index])) {
            return 0;
        }
    }
    for (uint32_t index = 0U; index < options->prefetch_rule_count; ++index) {
        if (!rule_valid(options->prefetch_rules[index])) {
            return 0;
        }
    }
    for (uint32_t alias = 0U; alias < context->residency->alias_count; ++alias) {
        if (context->alias_json_names[alias] == NULL) {
            return 0;
        }
    }
    for (uint32_t task = 0U; task < context->simulation->task_count; ++task) {
        if (context->task_json_names[task] == NULL) {
            return 0;
        }
    }
    return 1;
}

static int simulation_workspace_create(
    const ShadowSpillPressureFitContext *context,
    uint32_t action_capacity,
    SimulationWorkspace *workspace
) {
    memset(workspace, 0, sizeof(*workspace));
    workspace->task_capacity = context->simulation->task_count;
    workspace->transfer_capacity = action_capacity;
    workspace->device_capacity = context->simulation->device_count;
    workspace->tasks = calloc(
        workspace->task_capacity == 0U ? 1U : workspace->task_capacity,
        sizeof(*workspace->tasks)
    );
    workspace->transfers = calloc(
        workspace->transfer_capacity == 0U ? 1U : workspace->transfer_capacity,
        sizeof(*workspace->transfers)
    );
    workspace->peaks = calloc(
        workspace->device_capacity == 0U ? 1U : workspace->device_capacity,
        sizeof(*workspace->peaks)
    );
    if (workspace->tasks == NULL || workspace->transfers == NULL ||
        workspace->peaks == NULL) {
        free(workspace->tasks);
        free(workspace->transfers);
        free(workspace->peaks);
        memset(workspace, 0, sizeof(*workspace));
        return -1;
    }
    return 0;
}

static void simulation_workspace_destroy(SimulationWorkspace *workspace) {
    free(workspace->tasks);
    free(workspace->transfers);
    free(workspace->peaks);
    memset(workspace, 0, sizeof(*workspace));
}

static ShadowSpillSimulationStatus simulate_schedule(
    const ShadowSpillPressureFitContext *context,
    const ShadowSpillDenseSchedule *schedule,
    SimulationWorkspace *workspace,
    ShadowSpillSimulationResult *result
) {
    ShadowSpillSimulationProgram program;
    shadowspill_bind_dense_schedule(context->simulation, schedule, &program);
    *result = (ShadowSpillSimulationResult){
        .task_intervals = workspace->tasks,
        .task_interval_capacity = workspace->task_capacity,
        .transfer_intervals = workspace->transfers,
        .transfer_interval_capacity = workspace->transfer_capacity,
        .device_peaks = workspace->peaks,
        .device_peak_capacity = workspace->device_capacity,
    };
    return shadowspill_simulate(&program, result);
}

static int candidate_workspace_create(
    const ShadowSpillPressureFitContext *context,
    CandidateWorkspace *workspace
) {
    memset(workspace, 0, sizeof(*workspace));
    uint64_t cell_count = (uint64_t)context->residency->alias_count *
        context->residency->boundary_count;
    uint64_t pressure_count = (uint64_t)context->residency->device_count *
        context->residency->boundary_count;
    if (cell_count > SIZE_MAX || pressure_count > SIZE_MAX) {
        return -1;
    }
    size_t cells = (size_t)cell_count;
    workspace->residency_cache.cell_count = cells;
    workspace->residency_cache.pressure_count = (size_t)pressure_count;
    workspace->schedule_cache.cell_count = cells;
    workspace->resident = calloc(cells == 0U ? 1U : cells, 1U);
    workspace->breaks = calloc(cells == 0U ? 1U : cells, 1U);
    workspace->base_resident = calloc(cells == 0U ? 1U : cells, 1U);
    workspace->base_breaks = calloc(cells == 0U ? 1U : cells, 1U);
    workspace->extra_pressure = calloc(
        pressure_count == 0U ? 1U : (size_t)pressure_count,
        sizeof(*workspace->extra_pressure)
    );
    if (workspace->resident == NULL || workspace->breaks == NULL ||
        workspace->base_resident == NULL || workspace->base_breaks == NULL ||
        workspace->extra_pressure == NULL ||
        shadowspill_schedule_storage_create(
            context->residency->alias_count,
            context->simulation->task_count,
            &workspace->schedule
        ) != 0 ||
        shadowspill_schedule_storage_create(
            context->residency->alias_count,
            context->simulation->task_count,
            &workspace->selected
        ) != 0 ||
        simulation_workspace_create(
            context,
            workspace->schedule.action_capacity,
            &workspace->simulation
        ) != 0 ||
        shadowspill_residency_workspace_create(
            context->residency,
            &workspace->residency_workspace
        ) != 0) {
        return -1;
    }
    return 0;
}

static void candidate_workspace_destroy(CandidateWorkspace *workspace) {
    if (workspace == NULL) {
        return;
    }
    free(workspace->resident);
    free(workspace->breaks);
    free(workspace->base_resident);
    free(workspace->base_breaks);
    free(workspace->extra_pressure);
    shadowspill_schedule_storage_destroy(&workspace->schedule);
    shadowspill_schedule_storage_destroy(&workspace->selected);
    simulation_workspace_destroy(&workspace->simulation);
    shadowspill_residency_workspace_destroy(workspace->residency_workspace);
    for (uint32_t index = 0U; index < workspace->residency_cache.count; ++index) {
        free(workspace->residency_cache.entries[index].extra_pressure);
        free(workspace->residency_cache.entries[index].resident);
        free(workspace->residency_cache.entries[index].breaks);
    }
    free(workspace->residency_cache.entries);
    free(workspace->residency_cache.index.slots);
    for (uint32_t index = 0U; index < workspace->schedule_cache.count; ++index) {
        ScheduleCacheEntry *entry = &workspace->schedule_cache.entries[index];
        free(entry->resident);
        free(entry->breaks);
        free(entry->schedule.action_trigger_tasks);
        free(entry->schedule.action_aliases);
        free(entry->schedule.action_kinds);
        free(entry->schedule.initial_aliases);
        free(entry->schedule.initial_locations);
        free(entry->schedule.final_aliases);
        free(entry->schedule.final_locations);
    }
    free(workspace->schedule_cache.entries);
    free(workspace->schedule_cache.index.slots);
    for (uint32_t index = 0U; index < workspace->simulation_cache.count; ++index) {
        ShadowSpillDenseSchedule *schedule =
            &workspace->simulation_cache.entries[index].schedule;
        free(schedule->action_trigger_tasks);
        free(schedule->action_aliases);
        free(schedule->action_kinds);
        free(schedule->initial_aliases);
        free(schedule->initial_locations);
        free(schedule->final_aliases);
        free(schedule->final_locations);
    }
    free(workspace->simulation_cache.entries);
    free(workspace->simulation_cache.index.slots);
    memset(workspace, 0, sizeof(*workspace));
}

static void residency_options(
    const ShadowSpillPressureFitContext *context,
    CandidateWorkspace *workspace,
    uint8_t strategy,
    ShadowSpillResidencyOptions *options
) {
    *options = (ShadowSpillResidencyOptions){
        .minimize_transfer =
            strategy == SHADOWSPILL_RESIDENCY_HEADROOM_TRANSFER ||
                strategy == SHADOWSPILL_RESIDENCY_TIGHT_TRANSFER
            ? 1U
            : 0U,
        .prefetch_headroom =
            strategy == SHADOWSPILL_RESIDENCY_HEADROOM_STALL ||
                strategy == SHADOWSPILL_RESIDENCY_HEADROOM_TRANSFER
            ? 1U
            : 0U,
        .seed_resident = context->seed_resident,
        .seed_breaks = context->seed_breaks,
        .extra_pressure_bytes = workspace->extra_pressure,
    };
}

static ShadowSpillPlannerStatus reduce(
    const ShadowSpillPressureFitContext *context,
    CandidateWorkspace *workspace,
    const ShadowSpillResidencyOptions *options,
    uint8_t *resident,
    uint8_t *breaks,
    ShadowSpillResidencyResult *result
) {
    uint64_t cells = (uint64_t)context->residency->alias_count *
        context->residency->boundary_count;
    *result = (ShadowSpillResidencyResult){
        .resident = resident,
        .resident_capacity = cells,
        .breaks = breaks,
        .break_capacity = cells,
    };
    return shadowspill_reduce_residency_reusing(
        context->residency,
        options,
        result,
        workspace->residency_workspace
    );
}

static ResidencyCacheEntry *find_residency_cache(
    CandidateWorkspace *workspace,
    uint8_t strategy,
    uint64_t hash
) {
    ResidencyCache *cache = &workspace->residency_cache;
    size_t bytes = cache->pressure_count * sizeof(*workspace->extra_pressure);
    uint32_t slot = hash_index_start(&cache->index, hash);
    while (slot != UINT32_MAX &&
           cache->index.slots[slot].entry_plus_one != 0U) {
        HashSlot indexed = cache->index.slots[slot];
        ResidencyCacheEntry *entry =
            &cache->entries[indexed.entry_plus_one - 1U];
        if (indexed.hash == hash && entry->strategy == strategy &&
            memcmp(entry->extra_pressure, workspace->extra_pressure, bytes) == 0) {
            return entry;
        }
        slot = hash_index_next(&cache->index, slot);
    }
    return NULL;
}

static ResidencyCacheEntry *append_residency_cache(
    CandidateWorkspace *workspace,
    uint8_t strategy,
    uint64_t hash,
    ShadowSpillPlannerStatus status,
    const ShadowSpillResidencyResult *result,
    const uint8_t *resident,
    const uint8_t *breaks
) {
    ResidencyCache *cache = &workspace->residency_cache;
    if (cache->count == cache->capacity) {
        uint32_t capacity = cache->capacity == 0U ? 16U : cache->capacity * 2U;
        if (capacity < cache->capacity) {
            return NULL;
        }
        ResidencyCacheEntry *entries = realloc(
            cache->entries,
            (size_t)capacity * sizeof(*entries)
        );
        if (entries == NULL) {
            return NULL;
        }
        memset(
            entries + cache->capacity,
            0,
            (size_t)(capacity - cache->capacity) * sizeof(*entries)
        );
        cache->entries = entries;
        cache->capacity = capacity;
    }
    ResidencyCacheEntry *entry = &cache->entries[cache->count];
    entry->extra_pressure = malloc(
        (cache->pressure_count == 0U ? 1U : cache->pressure_count) *
        sizeof(*entry->extra_pressure)
    );
    entry->resident = malloc(cache->cell_count == 0U ? 1U : cache->cell_count);
    entry->breaks = malloc(cache->cell_count == 0U ? 1U : cache->cell_count);
    if (entry->extra_pressure == NULL || entry->resident == NULL ||
        entry->breaks == NULL) {
        free(entry->extra_pressure);
        free(entry->resident);
        free(entry->breaks);
        memset(entry, 0, sizeof(*entry));
        return NULL;
    }
    memcpy(
        entry->extra_pressure,
        workspace->extra_pressure,
        cache->pressure_count * sizeof(*entry->extra_pressure)
    );
    memcpy(entry->resident, resident, cache->cell_count);
    memcpy(entry->breaks, breaks, cache->cell_count);
    entry->strategy = strategy;
    entry->hash = hash;
    entry->status = status;
    entry->error_device = result->error_device;
    entry->error_boundary = result->error_boundary;
    entry->required_bytes = result->required_bytes;
    entry->capacity_bytes = result->capacity_bytes;
    if (hash_index_insert(&cache->index, hash, cache->count) != 0) {
        free(entry->extra_pressure);
        free(entry->resident);
        free(entry->breaks);
        memset(entry, 0, sizeof(*entry));
        return NULL;
    }
    ++cache->count;
    return entry;
}

static ShadowSpillPlannerStatus reduce_cached(
    const ShadowSpillPressureFitContext *context,
    CandidateWorkspace *workspace,
    const ShadowSpillResidencyOptions *options,
    uint8_t strategy,
    uint8_t *resident,
    uint8_t *breaks,
    ShadowSpillResidencyResult *result
) {
    ResidencyCache *cache = &workspace->residency_cache;
    uint64_t hash = residency_cache_hash(workspace, strategy);
    ResidencyCacheEntry *entry = find_residency_cache(workspace, strategy, hash);
    if (entry == NULL) {
        ++workspace->residency_cache_misses;
        ShadowSpillResidencyResult computed;
        ShadowSpillPlannerStatus status = reduce(
            context,
            workspace,
            options,
            resident,
            breaks,
            &computed
        );
        entry = append_residency_cache(
            workspace,
            strategy,
            hash,
            status,
            &computed,
            resident,
            breaks
        );
        if (entry == NULL) {
            return SHADOWSPILL_PLANNER_ALLOCATION_FAILURE;
        }
    } else {
        ++workspace->residency_cache_hits;
    }
    memcpy(resident, entry->resident, cache->cell_count);
    memcpy(breaks, entry->breaks, cache->cell_count);
    *result = (ShadowSpillResidencyResult){
        .status = (uint32_t)entry->status,
        .error_device = entry->error_device,
        .error_boundary = entry->error_boundary,
        .required_bytes = entry->required_bytes,
        .capacity_bytes = entry->capacity_bytes,
        .resident = resident,
        .resident_capacity = cache->cell_count,
        .breaks = breaks,
        .break_capacity = cache->cell_count,
    };
    return entry->status;
}

static ScheduleCacheEntry *find_schedule_cache(
    CandidateWorkspace *workspace,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint8_t rule,
    uint8_t coalesced,
    uint8_t prefetch_headroom,
    uint64_t hash
) {
    ScheduleCache *cache = &workspace->schedule_cache;
    uint32_t slot = hash_index_start(&cache->index, hash);
    while (slot != UINT32_MAX &&
           cache->index.slots[slot].entry_plus_one != 0U) {
        HashSlot indexed = cache->index.slots[slot];
        ScheduleCacheEntry *entry =
            &cache->entries[indexed.entry_plus_one - 1U];
        if (indexed.hash == hash && entry->rule == rule &&
            entry->coalesced == coalesced &&
            entry->prefetch_headroom == prefetch_headroom &&
            memcmp(entry->resident, resident, cache->cell_count) == 0 &&
            memcmp(entry->breaks, breaks, cache->cell_count) == 0) {
            return entry;
        }
        slot = hash_index_next(&cache->index, slot);
    }
    return NULL;
}

static int clone_dense_schedule(
    const ShadowSpillDenseSchedule *source,
    ShadowSpillDenseSchedule *destination
) {
    memset(destination, 0, sizeof(*destination));
    uint32_t actions = source->action_count == 0U ? 1U : source->action_count;
    uint32_t initial = source->initial_count == 0U ? 1U : source->initial_count;
    uint32_t final = source->final_count == 0U ? 1U : source->final_count;
    destination->action_trigger_tasks = malloc(
        (size_t)actions * sizeof(*destination->action_trigger_tasks)
    );
    destination->action_aliases = malloc(
        (size_t)actions * sizeof(*destination->action_aliases)
    );
    destination->action_kinds = malloc(
        (size_t)actions * sizeof(*destination->action_kinds)
    );
    destination->initial_aliases = malloc(
        (size_t)initial * sizeof(*destination->initial_aliases)
    );
    destination->initial_locations = malloc(
        (size_t)initial * sizeof(*destination->initial_locations)
    );
    destination->final_aliases = malloc(
        (size_t)final * sizeof(*destination->final_aliases)
    );
    destination->final_locations = malloc(
        (size_t)final * sizeof(*destination->final_locations)
    );
    if (destination->action_trigger_tasks == NULL ||
        destination->action_aliases == NULL || destination->action_kinds == NULL ||
        destination->initial_aliases == NULL ||
        destination->initial_locations == NULL ||
        destination->final_aliases == NULL ||
        destination->final_locations == NULL) {
        free(destination->action_trigger_tasks);
        free(destination->action_aliases);
        free(destination->action_kinds);
        free(destination->initial_aliases);
        free(destination->initial_locations);
        free(destination->final_aliases);
        free(destination->final_locations);
        memset(destination, 0, sizeof(*destination));
        return -1;
    }
    destination->action_count = source->action_count;
    destination->initial_count = source->initial_count;
    destination->final_count = source->final_count;
    memcpy(
        destination->action_trigger_tasks,
        source->action_trigger_tasks,
        (size_t)source->action_count * sizeof(*source->action_trigger_tasks)
    );
    memcpy(
        destination->action_aliases,
        source->action_aliases,
        (size_t)source->action_count * sizeof(*source->action_aliases)
    );
    memcpy(
        destination->action_kinds,
        source->action_kinds,
        (size_t)source->action_count * sizeof(*source->action_kinds)
    );
    memcpy(
        destination->initial_aliases,
        source->initial_aliases,
        (size_t)source->initial_count * sizeof(*source->initial_aliases)
    );
    memcpy(
        destination->initial_locations,
        source->initial_locations,
        (size_t)source->initial_count * sizeof(*source->initial_locations)
    );
    memcpy(
        destination->final_aliases,
        source->final_aliases,
        (size_t)source->final_count * sizeof(*source->final_aliases)
    );
    memcpy(
        destination->final_locations,
        source->final_locations,
        (size_t)source->final_count * sizeof(*source->final_locations)
    );
    return 0;
}

static int dense_schedule_equal(
    const ShadowSpillDenseSchedule *left,
    const ShadowSpillDenseSchedule *right
) {
    return left->action_count == right->action_count &&
        left->initial_count == right->initial_count &&
        left->final_count == right->final_count &&
        memcmp(
            left->action_trigger_tasks,
            right->action_trigger_tasks,
            (size_t)left->action_count * sizeof(*left->action_trigger_tasks)
        ) == 0 &&
        memcmp(
            left->action_aliases,
            right->action_aliases,
            (size_t)left->action_count * sizeof(*left->action_aliases)
        ) == 0 &&
        memcmp(
            left->action_kinds,
            right->action_kinds,
            (size_t)left->action_count * sizeof(*left->action_kinds)
        ) == 0 &&
        memcmp(
            left->initial_aliases,
            right->initial_aliases,
            (size_t)left->initial_count * sizeof(*left->initial_aliases)
        ) == 0 &&
        memcmp(
            left->initial_locations,
            right->initial_locations,
            (size_t)left->initial_count * sizeof(*left->initial_locations)
        ) == 0 &&
        memcmp(
            left->final_aliases,
            right->final_aliases,
            (size_t)left->final_count * sizeof(*left->final_aliases)
        ) == 0 &&
        memcmp(
            left->final_locations,
            right->final_locations,
            (size_t)left->final_count * sizeof(*left->final_locations)
        ) == 0;
}

static int append_simulation_cache(
    CandidateWorkspace *workspace,
    const ShadowSpillSimulationResult *result,
    uint64_t hash
) {
    SimulationCache *cache = &workspace->simulation_cache;
    if (cache->count == cache->capacity) {
        uint32_t capacity = cache->capacity == 0U ? 16U : cache->capacity * 2U;
        if (capacity < cache->capacity) {
            return -1;
        }
        SimulationCacheEntry *entries = realloc(
            cache->entries,
            (size_t)capacity * sizeof(*entries)
        );
        if (entries == NULL) {
            return -1;
        }
        memset(
            entries + cache->capacity,
            0,
            (size_t)(capacity - cache->capacity) * sizeof(*entries)
        );
        cache->entries = entries;
        cache->capacity = capacity;
    }
    SimulationCacheEntry *entry = &cache->entries[cache->count];
    if (clone_dense_schedule(&workspace->schedule.value, &entry->schedule) != 0) {
        return -1;
    }
    entry->result = *result;
    entry->hash = hash;
    entry->result.task_intervals = NULL;
    entry->result.transfer_intervals = NULL;
    entry->result.device_peaks = NULL;
    entry->result.task_interval_capacity = 0U;
    entry->result.transfer_interval_capacity = 0U;
    entry->result.device_peak_capacity = 0U;
    if (hash_index_insert(&cache->index, hash, cache->count) != 0) {
        free(entry->schedule.action_trigger_tasks);
        free(entry->schedule.action_aliases);
        free(entry->schedule.action_kinds);
        free(entry->schedule.initial_aliases);
        free(entry->schedule.initial_locations);
        free(entry->schedule.final_aliases);
        free(entry->schedule.final_locations);
        memset(entry, 0, sizeof(*entry));
        return -1;
    }
    ++cache->count;
    return 0;
}

static int simulate_cached(
    const ShadowSpillPressureFitContext *context,
    CandidateWorkspace *workspace,
    ShadowSpillSimulationResult *result
) {
    SimulationCache *cache = &workspace->simulation_cache;
    uint64_t hash = dense_schedule_hash(&workspace->schedule.value);
    uint32_t slot = hash_index_start(&cache->index, hash);
    while (slot != UINT32_MAX &&
           cache->index.slots[slot].entry_plus_one != 0U) {
        HashSlot indexed = cache->index.slots[slot];
        SimulationCacheEntry *entry =
            &cache->entries[indexed.entry_plus_one - 1U];
        if (indexed.hash == hash && dense_schedule_equal(
                &entry->schedule,
                &workspace->schedule.value
            )) {
            *result = entry->result;
            ++workspace->simulation_cache_hits;
            return 0;
        }
        slot = hash_index_next(&cache->index, slot);
    }
    (void)simulate_schedule(
        context,
        &workspace->schedule.value,
        &workspace->simulation,
        result
    );
    ++workspace->simulation_calls;
    return append_simulation_cache(workspace, result, hash);
}

static ScheduleCacheEntry *append_schedule_cache(
    CandidateWorkspace *workspace,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint8_t rule,
    uint8_t coalesced,
    uint8_t prefetch_headroom,
    uint64_t hash
) {
    ScheduleCache *cache = &workspace->schedule_cache;
    if (cache->count == cache->capacity) {
        uint32_t capacity = cache->capacity == 0U ? 16U : cache->capacity * 2U;
        if (capacity < cache->capacity) {
            return NULL;
        }
        ScheduleCacheEntry *entries = realloc(
            cache->entries,
            (size_t)capacity * sizeof(*entries)
        );
        if (entries == NULL) {
            return NULL;
        }
        memset(
            entries + cache->capacity,
            0,
            (size_t)(capacity - cache->capacity) * sizeof(*entries)
        );
        cache->entries = entries;
        cache->capacity = capacity;
    }
    ScheduleCacheEntry *entry = &cache->entries[cache->count];
    entry->resident = malloc(cache->cell_count == 0U ? 1U : cache->cell_count);
    entry->breaks = malloc(cache->cell_count == 0U ? 1U : cache->cell_count);
    if (entry->resident == NULL || entry->breaks == NULL ||
        clone_dense_schedule(&workspace->schedule.value, &entry->schedule) != 0) {
        free(entry->resident);
        free(entry->breaks);
        memset(entry, 0, sizeof(*entry));
        return NULL;
    }
    memcpy(entry->resident, resident, cache->cell_count);
    memcpy(entry->breaks, breaks, cache->cell_count);
    entry->rule = rule;
    entry->hash = hash;
    entry->coalesced = coalesced;
    entry->prefetch_headroom = prefetch_headroom;
    if (hash_index_insert(&cache->index, hash, cache->count) != 0) {
        free(entry->resident);
        free(entry->breaks);
        free(entry->schedule.action_trigger_tasks);
        free(entry->schedule.action_aliases);
        free(entry->schedule.action_kinds);
        free(entry->schedule.initial_aliases);
        free(entry->schedule.initial_locations);
        free(entry->schedule.final_aliases);
        free(entry->schedule.final_locations);
        memset(entry, 0, sizeof(*entry));
        return NULL;
    }
    ++cache->count;
    return entry;
}

static int emit_cached(
    const ShadowSpillScheduleFacts *facts,
    CandidateWorkspace *workspace,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint8_t rule,
    uint8_t coalesced,
    uint8_t prefetch_headroom
) {
    uint64_t hash = schedule_cache_hash(
        workspace,
        resident,
        breaks,
        rule,
        coalesced,
        prefetch_headroom
    );
    ScheduleCacheEntry *entry = find_schedule_cache(
        workspace,
        resident,
        breaks,
        rule,
        coalesced,
        prefetch_headroom,
        hash
    );
    if (entry != NULL) {
        ++workspace->schedule_cache_hits;
        ShadowSpillScheduleStorage source = {
            .value = entry->schedule,
            .action_capacity = entry->schedule.action_count,
            .initial_capacity = entry->schedule.initial_count,
            .final_capacity = entry->schedule.final_count,
        };
        return shadowspill_schedule_storage_copy(&workspace->schedule, &source);
    }
    if (shadowspill_emit_dense_schedule(
            facts,
            resident,
            breaks,
            rule,
            coalesced != 0U,
            prefetch_headroom != 0U,
            &workspace->schedule
        ) != 0) {
        return -1;
    }
    ++workspace->schedule_emissions;
    return append_schedule_cache(
        workspace,
        resident,
        breaks,
        rule,
        coalesced,
        prefetch_headroom,
        hash
    ) == NULL
        ? -1
        : 0;
}

static void copy_simulation_error(
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic,
    const ShadowSpillSimulationResult *simulation
) {
    diagnostic->simulation_status = simulation->status;
    diagnostic->error_task = simulation->error_task;
    diagnostic->error_alias = simulation->error_alias;
    diagnostic->error_device = simulation->error_device;
    diagnostic->error_location = simulation->error_location;
    diagnostic->error_time_ns = simulation->error_time_ns;
    diagnostic->error_capacity_bytes = simulation->error_capacity_bytes;
    diagnostic->error_used_bytes = simulation->error_used_bytes;
    diagnostic->error_requested_bytes = simulation->error_requested_bytes;
}

static void copy_analytic_error(
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic,
    const ShadowSpillResidencyResult *residency
) {
    diagnostic->status = SHADOWSPILL_CANDIDATE_ANALYTIC_INFEASIBLE;
    diagnostic->error_device = residency->error_device;
    diagnostic->error_boundary = residency->error_boundary;
    diagnostic->error_required_bytes = residency->required_bytes;
    diagnostic->error_capacity_bytes = residency->capacity_bytes;
}

static int add_repair_pressure(
    const ShadowSpillPressureFitContext *context,
    CandidateWorkspace *workspace,
    const ShadowSpillSimulationResult *failure
) {
    if (failure->status != SHADOWSPILL_SIMULATION_INITIAL_DEVICE_CAPACITY &&
        failure->status != SHADOWSPILL_SIMULATION_PREFETCH_DEVICE_CAPACITY &&
        failure->status != SHADOWSPILL_SIMULATION_TASK_DEVICE_CAPACITY) {
        return 0;
    }
    if (failure->error_device == SHADOWSPILL_SIMULATOR_NO_INDEX ||
        failure->error_device >= context->residency->device_count) {
        return 0;
    }
    int32_t boundary = -1;
    if (failure->status == SHADOWSPILL_SIMULATION_TASK_DEVICE_CAPACITY) {
        if (failure->error_task == SHADOWSPILL_SIMULATOR_NO_INDEX) {
            return 0;
        }
        boundary = (int32_t)failure->error_task - 1;
    } else if (failure->status ==
               SHADOWSPILL_SIMULATION_PREFETCH_DEVICE_CAPACITY) {
        if (failure->error_task == SHADOWSPILL_SIMULATOR_NO_INDEX) {
            return 0;
        }
        boundary = (int32_t)failure->error_task;
    }
    uint64_t capacity = failure->error_capacity_bytes != 0U
        ? failure->error_capacity_bytes
        : context->residency->device_capacity_bytes[failure->error_device];
    uint64_t total = failure->error_used_bytes;
    if (failure->error_requested_bytes > UINT64_MAX - total) {
        total = UINT64_MAX;
    } else {
        total += failure->error_requested_bytes;
    }
    uint64_t excess = total > capacity ? total - capacity : 1U;
    uint32_t index = (uint32_t)(boundary + 1);
    uint64_t position =
        (uint64_t)failure->error_device * context->residency->boundary_count +
        index;
    if (workspace->extra_pressure[position] > UINT64_MAX - excess) {
        workspace->extra_pressure[position] = UINT64_MAX;
    } else {
        workspace->extra_pressure[position] += excess;
    }
    return 1;
}

static void initialize_diagnostic(
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic,
    uint8_t strategy,
    uint8_t rule,
    uint8_t coalesced
) {
    memset(diagnostic, 0, sizeof(*diagnostic));
    diagnostic->status = SHADOWSPILL_CANDIDATE_INTERNAL_ERROR;
    diagnostic->residency_strategy = strategy;
    diagnostic->prefetch_rule = rule;
    diagnostic->coalesced = coalesced;
    diagnostic->simulation_status = SHADOWSPILL_SIMULATION_OK;
    diagnostic->error_task = SHADOWSPILL_SIMULATOR_NO_INDEX;
    diagnostic->error_alias = SHADOWSPILL_SIMULATOR_NO_INDEX;
    diagnostic->error_device = SHADOWSPILL_SIMULATOR_NO_INDEX;
    diagnostic->error_boundary = INT32_MIN;
}

static int evaluate_candidate(
    const ShadowSpillPressureFitContext *context,
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillPressureFitContextOptions *portfolio_options,
    CandidateWorkspace *workspace,
    uint8_t strategy,
    uint8_t rule,
    uint8_t coalesced,
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic
) {
    initialize_diagnostic(diagnostic, strategy, rule, coalesced);
    uint64_t cells = (uint64_t)context->residency->alias_count *
        context->residency->boundary_count;
    uint64_t pressure_cells = (uint64_t)context->residency->device_count *
        context->residency->boundary_count;
    memset(
        workspace->extra_pressure,
        0,
        (size_t)pressure_cells * sizeof(*workspace->extra_pressure)
    );
    memcpy(workspace->resident, workspace->base_resident, (size_t)cells);
    memcpy(workspace->breaks, workspace->base_breaks, (size_t)cells);

    ShadowSpillResidencyOptions reduce_options;
    residency_options(context, workspace, strategy, &reduce_options);
    int need_emit = 1;
    while (1) {
        if (need_emit != 0) {
            uint64_t schedule_started = monotonic_time_ns();
            if (rule == SHADOWSPILL_PREFETCH_INTERVAL_ENTRY &&
                shadowspill_extend_interval_entries(
                    facts,
                    workspace->resident,
                    workspace->breaks
                ) != 0) {
                return -1;
            }
            int emitted = emit_cached(
                facts,
                workspace,
                workspace->resident,
                workspace->breaks,
                rule,
                coalesced,
                reduce_options.prefetch_headroom
            );
            if (emitted != 0) {
                return -1;
            }
            workspace->schedule_time_ns +=
                monotonic_time_ns() - schedule_started;
            need_emit = 0;
        }

        ShadowSpillSimulationResult simulation;
        uint64_t simulation_started = monotonic_time_ns();
        if (simulate_cached(context, workspace, &simulation) != 0) {
            return -1;
        }
        workspace->simulation_time_ns +=
            monotonic_time_ns() - simulation_started;
        ShadowSpillSimulationStatus simulation_status =
            (ShadowSpillSimulationStatus)simulation.status;
        if (simulation_status == SHADOWSPILL_SIMULATION_OK) {
            diagnostic->status = SHADOWSPILL_CANDIDATE_VALID;
            diagnostic->makespan_ns = simulation.makespan_ns;
            uint64_t digest_started = monotonic_time_ns();
            shadowspill_schedule_digest(
                context,
                &workspace->schedule.value,
                diagnostic->schedule_digest
            );
            workspace->digest_time_ns += monotonic_time_ns() - digest_started;
            return 1;
        }

        if (diagnostic->repair_attempts <
            portfolio_options->max_repair_attempts) {
            int delayed = shadowspill_delay_dense_prefetch(
                facts,
                &simulation,
                &workspace->schedule
            );
            if (delayed < 0) {
                return -1;
            }
            if (delayed > 0) {
                ++diagnostic->repair_attempts;
                continue;
            }
            if (add_repair_pressure(context, workspace, &simulation) != 0) {
                ++diagnostic->repair_attempts;
                ShadowSpillResidencyResult residency;
                uint64_t residency_started = monotonic_time_ns();
                ShadowSpillPlannerStatus status = reduce_cached(
                    context,
                    workspace,
                    &reduce_options,
                    strategy,
                    workspace->resident,
                    workspace->breaks,
                    &residency
                );
                workspace->residency_time_ns +=
                    monotonic_time_ns() - residency_started;
                if (status == SHADOWSPILL_PLANNER_ANALYTIC_INFEASIBLE) {
                    copy_analytic_error(diagnostic, &residency);
                    return 0;
                }
                if (status != SHADOWSPILL_PLANNER_OK) {
                    return -1;
                }
                need_emit = 1;
                continue;
            }
        }
        diagnostic->status = SHADOWSPILL_CANDIDATE_SIMULATION_INFEASIBLE;
        copy_simulation_error(diagnostic, &simulation);
        return 0;
    }
}

static int adopt_selected_schedule(
    ShadowSpillPressureFitContextResult *result,
    CandidateWorkspace *workspace
) {
    ShadowSpillDenseSchedule *source = &workspace->selected.value;
    ShadowSpillDenseSchedule *destination = &result->selected_schedule;
    *destination = *source;
    memset(source, 0, sizeof(*source));
    workspace->selected.action_capacity = 0U;
    workspace->selected.initial_capacity = 0U;
    workspace->selected.final_capacity = 0U;
    return 0;
}

void shadowspill_pressurefit_context_result_destroy(
    ShadowSpillPressureFitContextResult *result
) {
    if (result == NULL) {
        return;
    }
    free(result->candidates);
    free(result->selected_schedule.action_trigger_tasks);
    free(result->selected_schedule.action_aliases);
    free(result->selected_schedule.action_kinds);
    free(result->selected_schedule.initial_aliases);
    free(result->selected_schedule.initial_locations);
    free(result->selected_schedule.final_aliases);
    free(result->selected_schedule.final_locations);
    memset(result, 0, sizeof(*result));
}

ShadowSpillPlannerStatus shadowspill_evaluate_pressurefit_context(
    const ShadowSpillPressureFitContext *context,
    const ShadowSpillPressureFitContextOptions *options,
    ShadowSpillPressureFitContextResult *result
) {
    if (result == NULL) {
        return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    }
    memset(result, 0, sizeof(*result));
    result->selected_candidate_index = SHADOWSPILL_PLANNER_NO_INDEX;
    result->status = SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    if (!context_valid(context, options)) {
        return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    }

    uint32_t coalesced_count = options->evaluate_coalesced != 0U ? 2U : 1U;
    uint32_t candidate_count = 0U;
    if (multiply_u32(
            options->residency_strategy_count,
            options->prefetch_rule_count,
            &candidate_count
        ) != 0 ||
        multiply_u32(candidate_count, coalesced_count, &candidate_count) != 0) {
        return SHADOWSPILL_PLANNER_INVALID_ARGUMENT;
    }
    result->candidates = calloc(candidate_count, sizeof(*result->candidates));
    if (result->candidates == NULL) {
        result->status = SHADOWSPILL_PLANNER_ALLOCATION_FAILURE;
        return SHADOWSPILL_PLANNER_ALLOCATION_FAILURE;
    }
    result->candidate_count = candidate_count;

    ShadowSpillScheduleFacts facts = {0};
    CandidateWorkspace workspace = {0};
    if (shadowspill_schedule_facts_create(context, &facts) != 0 ||
        candidate_workspace_create(context, &workspace) != 0) {
        shadowspill_schedule_facts_destroy(&facts);
        candidate_workspace_destroy(&workspace);
        shadowspill_pressurefit_context_result_destroy(result);
        result->status = SHADOWSPILL_PLANNER_ALLOCATION_FAILURE;
        return SHADOWSPILL_PLANNER_ALLOCATION_FAILURE;
    }

    uint64_t pressure_cells = (uint64_t)context->residency->device_count *
        context->residency->boundary_count;
    uint32_t candidate_index = 0U;
    for (uint32_t strategy_index = 0U;
         strategy_index < options->residency_strategy_count;
         ++strategy_index) {
        uint8_t strategy = options->residency_strategies[strategy_index];
        memset(
            workspace.extra_pressure,
            0,
            (size_t)pressure_cells * sizeof(*workspace.extra_pressure)
        );
        ShadowSpillResidencyOptions reduce_options;
        residency_options(context, &workspace, strategy, &reduce_options);
        ShadowSpillResidencyResult base_result;
        uint64_t residency_started = monotonic_time_ns();
        ShadowSpillPlannerStatus base_status = reduce_cached(
            context,
            &workspace,
            &reduce_options,
            strategy,
            workspace.base_resident,
            workspace.base_breaks,
            &base_result
        );
        workspace.residency_time_ns += monotonic_time_ns() - residency_started;
        for (uint32_t rule_index = 0U; rule_index < options->prefetch_rule_count;
             ++rule_index) {
            uint8_t rule = options->prefetch_rules[rule_index];
            for (uint32_t coalesced = 0U; coalesced < coalesced_count;
                 ++coalesced) {
                ShadowSpillPressureFitCandidateDiagnostic *diagnostic =
                    &result->candidates[candidate_index];
                initialize_diagnostic(
                    diagnostic,
                    strategy,
                    rule,
                    (uint8_t)coalesced
                );
                if (base_status == SHADOWSPILL_PLANNER_ANALYTIC_INFEASIBLE) {
                    copy_analytic_error(diagnostic, &base_result);
                } else if (base_status != SHADOWSPILL_PLANNER_OK) {
                    result->status = SHADOWSPILL_PLANNER_INTERNAL_ERROR;
                    shadowspill_schedule_facts_destroy(&facts);
                    candidate_workspace_destroy(&workspace);
                    return SHADOWSPILL_PLANNER_INTERNAL_ERROR;
                } else {
                    int valid = evaluate_candidate(
                        context,
                        &facts,
                        options,
                        &workspace,
                        strategy,
                        rule,
                        (uint8_t)coalesced,
                        diagnostic
                    );
                    if (valid < 0) {
                        result->status = SHADOWSPILL_PLANNER_INTERNAL_ERROR;
                        shadowspill_schedule_facts_destroy(&facts);
                        candidate_workspace_destroy(&workspace);
                        return SHADOWSPILL_PLANNER_INTERNAL_ERROR;
                    }
                    if (valid > 0 &&
                        (result->selected_candidate_index ==
                             SHADOWSPILL_PLANNER_NO_INDEX ||
                         diagnostic->makespan_ns < result->selected_makespan_ns)) {
                        result->selected_candidate_index = candidate_index;
                        result->selected_makespan_ns = diagnostic->makespan_ns;
                        if (shadowspill_schedule_storage_copy(
                                &workspace.selected,
                                &workspace.schedule
                            ) != 0) {
                            result->status = SHADOWSPILL_PLANNER_INTERNAL_ERROR;
                            shadowspill_schedule_facts_destroy(&facts);
                            candidate_workspace_destroy(&workspace);
                            return SHADOWSPILL_PLANNER_INTERNAL_ERROR;
                        }
                    }
                }
                ++candidate_index;
            }
        }
    }

    ShadowSpillPlannerStatus status = SHADOWSPILL_PLANNER_OK;
    if (result->selected_candidate_index == SHADOWSPILL_PLANNER_NO_INDEX) {
        status = SHADOWSPILL_PLANNER_NO_FEASIBLE_CANDIDATE;
    } else {
        adopt_selected_schedule(result, &workspace);
    }
    result->status = status;
    result->residency_cache_hits = workspace.residency_cache_hits;
    result->residency_cache_misses = workspace.residency_cache_misses;
    result->schedule_emissions = workspace.schedule_emissions;
    result->schedule_cache_hits = workspace.schedule_cache_hits;
    result->simulation_calls = workspace.simulation_calls;
    result->simulation_cache_hits = workspace.simulation_cache_hits;
    result->residency_time_ns = workspace.residency_time_ns;
    result->schedule_time_ns = workspace.schedule_time_ns;
    result->simulation_time_ns = workspace.simulation_time_ns;
    result->digest_time_ns = workspace.digest_time_ns;
    shadowspill_schedule_facts_destroy(&facts);
    candidate_workspace_destroy(&workspace);
    return status;
}
