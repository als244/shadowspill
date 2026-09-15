/* Fingerprints, and the two caches that keep one schedule from being
 * emitted or simulated twice. */
#include "internal.h"

int shadowspill_candidate_fingerprint_equal(Fingerprint left, Fingerprint right) {
    return left.low == right.low && left.high == right.high;
}


static uint64_t hash_words(
    uint64_t hash, const void *data, size_t size, uint64_t prime
) {
    const uint8_t *bytes = data;
    size_t index = 0U;
    for (; index + 8U <= size; index += 8U) {
        uint64_t word;
        memcpy(&word, bytes + index, sizeof(word));
        hash = (hash ^ word) * prime;
    }
    for (; index < size; ++index) {
        hash = (hash ^ bytes[index]) * prime;
    }
    return hash;
}

uint64_t shadowspill_candidate_hash_bytes(uint64_t hash, const void *data, size_t size) {
    return hash_words(hash, data, size, UINT64_C(1099511628211));
}

uint64_t shadowspill_candidate_hash_bytes_high(uint64_t hash, const void *data, size_t size) {
    return hash_words(hash, data, size, UINT64_C(0x9E3779B97F4A7C15));
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

static uint64_t schedule_memo_hash(
    Fingerprint residency,
    uint8_t rule,
    uint8_t coalesced,
    uint8_t fetch_headroom
) {
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = shadowspill_candidate_hash_bytes(hash, &residency, sizeof(residency));
    hash = shadowspill_candidate_hash_bytes(hash, &rule, sizeof(rule));
    hash = shadowspill_candidate_hash_bytes(hash, &coalesced, sizeof(coalesced));
    return shadowspill_candidate_hash_bytes(hash, &fetch_headroom, sizeof(fetch_headroom));
}

static uint64_t schedule_hash_with(
    const ShadowSpillIndexedSchedule *schedule,
    uint64_t hash,
    uint64_t (*step)(uint64_t, const void *, size_t)
) {
    uint64_t count = schedule->action_count;
    hash = step(hash, &count, sizeof(count));
    hash = step(
        hash,
        schedule->action_trigger_tasks,
        (size_t)schedule->action_count * sizeof(*schedule->action_trigger_tasks)
    );
    hash = step(
        hash,
        schedule->action_aliases,
        (size_t)schedule->action_count * sizeof(*schedule->action_aliases)
    );
    hash = step(
        hash,
        schedule->action_kinds,
        (size_t)schedule->action_count * sizeof(*schedule->action_kinds)
    );
    count = schedule->initial_count;
    hash = step(hash, &count, sizeof(count));
    hash = step(
        hash,
        schedule->initial_aliases,
        (size_t)schedule->initial_count * sizeof(*schedule->initial_aliases)
    );
    hash = step(
        hash,
        schedule->initial_locations,
        (size_t)schedule->initial_count * sizeof(*schedule->initial_locations)
    );
    count = schedule->final_count;
    hash = step(hash, &count, sizeof(count));
    hash = step(
        hash,
        schedule->final_aliases,
        (size_t)schedule->final_count * sizeof(*schedule->final_aliases)
    );
    return step(
        hash,
        schedule->final_locations,
        (size_t)schedule->final_count * sizeof(*schedule->final_locations)
    );
}

static Fingerprint indexed_schedule_fingerprint(
    const ShadowSpillIndexedSchedule *schedule
) {
    return (Fingerprint){
        .low = schedule_hash_with(schedule, UINT64_C(1469598103934665603), shadowspill_candidate_hash_bytes),
        .high = schedule_hash_with(schedule, UINT64_C(1099511628211), shadowspill_candidate_hash_bytes_high),
    };
}

static ScheduleMemoEntry *find_schedule_memo(
    CandidateWorkspace *workspace,
    Fingerprint residency,
    uint8_t rule,
    uint8_t coalesced,
    uint8_t fetch_headroom,
    uint64_t hash
) {
    ScheduleMemo *cache = &workspace->schedule_memo;
    for (uint32_t index = 0U; index < cache->count; ++index) {
        ScheduleMemoEntry *entry = &cache->entries[index];
        if (entry->hash == hash && entry->rule == rule &&
            entry->coalesced == coalesced &&
            entry->fetch_headroom == fetch_headroom &&
            shadowspill_candidate_fingerprint_equal(entry->residency, residency)) {
            return entry;
        }
    }
    return NULL;
}

void shadowspill_candidate_free_indexed_schedule(ShadowSpillIndexedSchedule *schedule) {
    free(schedule->action_trigger_tasks);
    free(schedule->action_aliases);
    free(schedule->action_kinds);
    free(schedule->initial_aliases);
    free(schedule->initial_locations);
    free(schedule->final_aliases);
    free(schedule->final_locations);
    memset(schedule, 0, sizeof(*schedule));
}

static int clone_indexed_schedule(
    const ShadowSpillIndexedSchedule *source,
    ShadowSpillIndexedSchedule *destination
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

static SimulationMemoEntry *append_simulation_memo(
    CandidateWorkspace *workspace,
    const ShadowSpillSimulationResult *result,
    ShadowSpillStatus admission_status,
    const ShadowSpillAdmissionReplayResult *admission,
    Fingerprint identity
) {
    SimulationMemo *cache = &workspace->simulation_memo;
    if (cache->count == cache->capacity) {
        uint32_t capacity = cache->capacity == 0U ? 16U : cache->capacity * 2U;
        if (capacity < cache->capacity) {
            return NULL;
        }
        SimulationMemoEntry *entries = realloc(
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
    SimulationMemoEntry *entry = &cache->entries[cache->count];
    entry->identity = identity;
    entry->result = *result;
    entry->first_violation = workspace->first_violation;
    entry->admission_status = (uint32_t)admission_status;
    entry->admission = *admission;
    entry->admission.decisions = NULL;
    entry->admission.dependencies = NULL;
    entry->admission.live_leases = NULL;
    entry->admission.decision_capacity = 0U;
    entry->admission.dependency_capacity = 0U;
    entry->admission.live_lease_capacity = 0U;
    entry->admission.live_lease_count = 0U;
    entry->result.task_intervals = NULL;
    entry->result.transfer_intervals = NULL;
    entry->result.device_peaks = NULL;
    entry->result.task_interval_capacity = 0U;
    entry->result.transfer_interval_capacity = 0U;
    entry->result.device_peak_capacity = 0U;
    if (hash_index_insert(&cache->index, identity.low, cache->count) != 0) {
        memset(entry, 0, sizeof(*entry));
        return NULL;
    }
    ++cache->count;
    return entry;
}

int shadowspill_candidate_simulate_cached(
    const ShadowSpillPressureFitProblem *problem,
    CandidateWorkspace *workspace,
    ShadowSpillSimulationResult *result,
    ShadowSpillStatus *admission_status,
    ShadowSpillAdmissionReplayResult *admission_result,
    ShadowSpillAdmissionAnnotation *admission_error_annotation,
    SimulationMemoEntry **selected_entry
) {
    SimulationMemo *cache = &workspace->simulation_memo;
    Fingerprint identity = indexed_schedule_fingerprint(&workspace->schedule.value);
    uint32_t slot = hash_index_start(&cache->index, identity.low);
    while (slot != UINT32_MAX &&
           cache->index.slots[slot].entry_plus_one != 0U) {
        HashSlot indexed = cache->index.slots[slot];
        SimulationMemoEntry *entry =
            &cache->entries[indexed.entry_plus_one - 1U];
        if (indexed.hash == identity.low &&
            shadowspill_candidate_fingerprint_equal(entry->identity, identity)) {
            *result = entry->result;
            workspace->first_violation = entry->first_violation;
            result->capacity_violations = &workspace->first_violation;
            *admission_status =
                (ShadowSpillStatus)entry->admission_status;
            *admission_result = entry->admission;
            *admission_error_annotation = (ShadowSpillAdmissionAnnotation){0};
            *selected_entry = entry;
            ++workspace->simulation_cache_hits;
            return 0;
        }
        slot = hash_index_next(&cache->index, slot);
    }
    if (shadowspill_candidate_simulate_schedule(
            problem,
            &workspace->schedule.value,
            &workspace->simulation,
            &workspace->admission,
            &workspace->first_violation,
            result,
            admission_status,
            admission_result
        ) != 0) {
        return -1;
    }
    *admission_error_annotation = (ShadowSpillAdmissionAnnotation){0};
    if (*admission_status == SHADOWSPILL_STATUS_REPLAY_INFEASIBLE) {
        const uint64_t operation = admission_result->error_operation_index;
        if (operation >= workspace->admission.operation_capacity) {
            return -1;
        }
        *admission_error_annotation =
            workspace->admission.annotations[operation];
        *selected_entry = NULL;
        return 0;
    }
    if (*admission_status != SHADOWSPILL_STATUS_OK) {
        return -1;
    }
    ++workspace->simulation_calls;
    *selected_entry = append_simulation_memo(
        workspace,
        result,
        *admission_status,
        admission_result,
        identity
    );
    return *selected_entry == NULL ? -1 : 0;
}

static ScheduleMemoEntry *append_schedule_memo(
    CandidateWorkspace *workspace,
    Fingerprint residency,
    uint8_t rule,
    uint8_t coalesced,
    uint8_t fetch_headroom,
    uint64_t hash
) {
    ScheduleMemo *cache = &workspace->schedule_memo;
    ScheduleMemoEntry *entry;
    if (cache->count < SCHEDULE_MEMO_CAPACITY) {
        entry = &cache->entries[cache->count++];
    } else {
        entry = &cache->entries[cache->next];
        cache->next = (cache->next + 1U) % SCHEDULE_MEMO_CAPACITY;
        shadowspill_candidate_free_indexed_schedule(&entry->schedule);
    }
    memset(entry, 0, sizeof(*entry));
    if (clone_indexed_schedule(&workspace->schedule.value, &entry->schedule) != 0) {
        return NULL;
    }
    entry->residency = residency;
    entry->rule = rule;
    entry->hash = hash;
    entry->coalesced = coalesced;
    entry->fetch_headroom = fetch_headroom;
    return entry;
}

int shadowspill_candidate_emit_cached(
    const ShadowSpillScheduleFacts *facts,
    CandidateWorkspace *workspace,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint8_t rule,
    uint8_t coalesced,
    uint8_t fetch_headroom
) {
    uint64_t hash = schedule_memo_hash(
        workspace->current_residency,
        rule,
        coalesced,
        fetch_headroom
    );
    ScheduleMemoEntry *entry = find_schedule_memo(
        workspace,
        workspace->current_residency,
        rule,
        coalesced,
        fetch_headroom,
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
    if (shadowspill_emit_indexed_schedule(
            facts,
            resident,
            breaks,
            rule,
            coalesced != 0U,
            fetch_headroom != 0U,
            &workspace->schedule
        ) != 0) {
        return -1;
    }
    ++workspace->schedule_emissions;
    return append_schedule_memo(
        workspace,
        workspace->current_residency,
        rule,
        coalesced,
        fetch_headroom,
        hash
    ) == NULL
        ? -1
        : 0;
}

