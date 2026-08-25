
#include "admission/internal.h"
#include "../common/platform.h"
#include "internal.h"
#include "candidates_internal.h"
#include "residency_internal.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The capacity a candidate is planning at, which starts as the device's
 * and falls as the candidate gives back what its layout overran. */
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
    uint64_t content_hash;
    uint8_t minimize_transfer;
    uint8_t prefetch_headroom;
    uint64_t *extra_pressure;
    uint8_t *resident_bits;
    uint8_t *break_bits;
    ShadowSpillStatus status;
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
    size_t packed_cell_count;
    size_t pressure_count;
    HashIndex index;
} ResidencyCache;

typedef struct ScheduleCacheEntry {
    uint64_t hash;
    uint32_t residency_key;
    uint8_t rule;
    uint8_t coalesced;
    uint8_t prefetch_headroom;
    ShadowSpillIndexedSchedule schedule;
} ScheduleCacheEntry;

typedef struct ScheduleCache {
    ScheduleCacheEntry *entries;
    uint32_t count;
    uint32_t capacity;
    HashIndex index;
} ScheduleCache;

typedef struct SimulationCacheEntry {
    uint64_t hash;
    ShadowSpillIndexedSchedule schedule;
    ShadowSpillSimulationResult result;
    /* Held by value, because the buffer the simulator wrote it into belongs
     * to the workspace and the next candidate overwrites it. */
    ShadowSpillCapacityViolation first_violation;
    ShadowSpillAdmissionReplayResult admission;
    uint32_t admission_status;
    uint8_t digest[SHADOWSPILL_PLANNER_DIGEST_BYTES];
    uint8_t digest_valid;
} SimulationCacheEntry;

typedef struct SimulationCache {
    SimulationCacheEntry *entries;
    uint32_t count;
    uint32_t capacity;
    HashIndex index;
} SimulationCache;


/*
 * Buffers for deciding whether one plan can be placed in the execution pool.
 *
 * Placement runs on the plan a candidate currently holds, so a candidate
 * places many plans over its life and the sizes barely change between them.
 * The arrays are grown on demand and reused rather than allocated per plan.
 */
typedef struct PlacementWorkspace {
    ShadowSpillAdmissionOperations operations;
    uint64_t *lease_ids;
    uint64_t *dependency_ids;
    uint64_t *bytes;
    uint64_t *alignments;
    uint8_t *kinds;
    uint8_t *purposes;
    uint8_t *boundaries;
    uint32_t *indices;
    uint32_t *allocation_offsets;
    uint32_t *lease_aliases;
    uint64_t *lease_starts;
    uint64_t *lease_retires;
    ShadowSpillLeaseLifetime *lifetimes;
    ShadowSpillLeaseIdentity *identities;
    uint64_t *allocation_step_leases;
    uint64_t *alias_leases;
    uint64_t *offsets;
    uint32_t *dynamic_aliases;
    uint64_t operation_capacity;
    uint64_t lease_capacity;
    uint32_t alias_capacity;
    uint32_t allocation_slot_capacity;
} PlacementWorkspace;

typedef struct CandidateWorkspace {
    /* What this plan has given back: the bytes its layout overran. It
     * shapes the plan -- the reducer charges it through `extra_pressure`
     * and the emitter measures against it -- and is reported on the plan so
     * a reader can tell what it was built for. The simulator is deliberately
     * not one of its readers: the plan runs on the machine the caller
     * described, so that is the capacity it is timed at. */
    uint64_t plan_capacity_given_back;
    uint8_t *resident;
    uint8_t *breaks;
    uint8_t *base_resident;
    uint8_t *base_breaks;
    uint8_t *repair_resident;
    uint8_t *repair_breaks;
    uint8_t *removable_aliases;
    uint64_t *extra_pressure;
    ShadowSpillScheduleStorage schedule;
    ShadowSpillScheduleStorage selected;
    /* The best plan this candidate has reached, kept because repairing
     * past a success can make it worse before it makes it better. */
    ShadowSpillScheduleStorage best;
    SimulationWorkspace simulation;
    /* Where a plan first came up short, which is what repair aims at
     * when the plan simulates but waits for memory. */
    ShadowSpillCapacityViolation first_violation;
    /* Buffers for placing a plan physically. Grown on demand and reused,
     * because a candidate places many plans and the sizes barely move. */
    PlacementWorkspace placement;
    ShadowSpillCandidateAdmissionWorkspace admission;
    ResidencyCache residency_cache;
    ScheduleCache schedule_cache;
    SimulationCache simulation_cache;
    ShadowSpillResidencyWorkspace *residency_workspace;
    ShadowSpillPrefetchTriggerConstraint *prefetch_constraints;
    uint32_t prefetch_constraint_count;
    uint32_t prefetch_constraint_capacity;
    uint32_t current_residency_key;
    uint32_t base_residency_key;
    uint64_t residency_cache_hits;
    uint64_t residency_cache_misses;
    uint64_t schedule_emissions;
    uint64_t schedule_cache_hits;
    uint64_t simulation_calls;
    uint64_t simulation_cache_hits;
    /* Where the time went. Written only by the functions that orchestrate
     * the work, never by the work itself. */
    ShadowSpillPressureFitSectionTiming sections;
    /* Scratch the reducer appends its cuts to, when a trajectory is being
     * recorded. Drained into the candidate's record after each reduction. */
    uint32_t *cut_scratch;
    uint64_t cut_scratch_capacity;
    uint64_t cut_scratch_count;
} CandidateWorkspace;

/*
 * One section of an orchestrator's time.
 *
 * Opened and closed around a call, so the partition of a function's time is
 * written where that function orchestrates the work rather than inside the
 * work. A section may be opened against any sink; nesting is expressed by
 * which sink it is given, not by the helper.
 */
typedef struct Section {
    uint64_t started;
    uint64_t *sink;
} Section;

static Section section_open(uint64_t *sink) {
    return (Section){shadowspill_monotonic_ns(), sink};
}

static void section_close(Section section) {
    *section.sink += shadowspill_monotonic_ns() - section.started;
}

/* What the named sections did not claim. Reported so the parts add up. */
static void section_close_total(
    ShadowSpillPressureFitSectionTiming *timing,
    uint64_t started
) {
    timing->total_ns = shadowspill_monotonic_ns() - started;
    const uint64_t named = timing->prepare_ns + timing->setup_ns +
        timing->reduce_ns + timing->emit_ns + timing->simulate_ns +
        timing->repair_ns + timing->digest_ns + timing->place_ns +
        timing->select_ns + timing->teardown_ns;
    timing->residual_ns =
        timing->total_ns > named ? timing->total_ns - named : 0U;
}

/* Append one step of a candidate's descent, growing the record as needed. */
static int record_step(
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic,
    ShadowSpillPressureFitReductionStep step
) {
    if (diagnostic->step_count == diagnostic->step_capacity) {
        uint32_t capacity = diagnostic->step_capacity == 0U
            ? 32U
            : diagnostic->step_capacity * 2U;
        if (capacity < diagnostic->step_capacity) {
            return -1;
        }
        ShadowSpillPressureFitReductionStep *grown = realloc(
            diagnostic->steps, (size_t)capacity * sizeof(*grown)
        );
        if (grown == NULL) {
            return -1;
        }
        diagnostic->steps = grown;
        diagnostic->step_capacity = capacity;
    }
    diagnostic->steps[diagnostic->step_count++] = step;
    return 0;
}

/* Move the cuts a reduction reported into the candidate's flat record. */
static int drain_cuts(
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic,
    CandidateWorkspace *workspace
) {
    const uint64_t added = workspace->cut_scratch_count;
    workspace->cut_scratch_count = 0U;
    if (added == 0U) {
        return 0;
    }
    if (diagnostic->cut_count + added > diagnostic->cut_capacity) {
        uint32_t capacity = diagnostic->cut_capacity == 0U
            ? 256U
            : diagnostic->cut_capacity;
        while (diagnostic->cut_count + added > capacity) {
            const uint32_t grown = capacity * 2U;
            if (grown < capacity) {
                return -1;
            }
            capacity = grown;
        }
        uint32_t *aliases = realloc(
            diagnostic->cut_aliases, (size_t)capacity * sizeof(*aliases)
        );
        if (aliases == NULL) {
            return -1;
        }
        diagnostic->cut_aliases = aliases;
        diagnostic->cut_capacity = capacity;
    }
    memcpy(
        diagnostic->cut_aliases + diagnostic->cut_count,
        workspace->cut_scratch,
        (size_t)added * sizeof(*workspace->cut_scratch)
    );
    diagnostic->cut_count += (uint32_t)added;
    return 0;
}

/* Mark the last recorded step, for facts only known after it was taken. */
static void mark_last_step(
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic,
    uint32_t flags,
    uint64_t required_bytes
) {
    if (diagnostic->step_count == 0U) {
        return;
    }
    ShadowSpillPressureFitReductionStep *step =
        &diagnostic->steps[diagnostic->step_count - 1U];
    step->flags |= flags;
    if (required_bytes != 0U) {
        step->required_bytes = required_bytes;
    }
}

static uint64_t repair_total(
    const ShadowSpillPressureFitRepairDiagnostics *repairs
) {
    return repairs->admission_prefetch_advance_attempts +
        repairs->admission_prefetch_delay_attempts +
        repairs->admission_pressure_boundary_attempts +
        repairs->simulation_prefetch_delay_attempts +
        repairs->simulation_pressure_boundary_attempts;
}

static ShadowSpillPressureFitWorkDiagnostics workspace_work(
    const CandidateWorkspace *workspace
) {
    return (ShadowSpillPressureFitWorkDiagnostics){
        .residency_cache_hits = workspace->residency_cache_hits,
        .residency_cache_misses = workspace->residency_cache_misses,
        .schedule_emissions = workspace->schedule_emissions,
        .schedule_cache_hits = workspace->schedule_cache_hits,
        .simulation_calls = workspace->simulation_calls,
        .simulation_cache_hits = workspace->simulation_cache_hits,
        .admission_calls = workspace->admission.calls,
        .sections = workspace->sections,
    };
}

static ShadowSpillPressureFitSectionTiming section_delta(
    ShadowSpillPressureFitSectionTiming after,
    ShadowSpillPressureFitSectionTiming before
) {
    return (ShadowSpillPressureFitSectionTiming){
        .prepare_ns = after.prepare_ns - before.prepare_ns,
        .setup_ns = after.setup_ns - before.setup_ns,
        .reduce_ns = after.reduce_ns - before.reduce_ns,
        .emit_ns = after.emit_ns - before.emit_ns,
        .simulate_ns = after.simulate_ns - before.simulate_ns,
        .repair_ns = after.repair_ns - before.repair_ns,
        .digest_ns = after.digest_ns - before.digest_ns,
        .place_ns = after.place_ns - before.place_ns,
        .select_ns = after.select_ns - before.select_ns,
        .teardown_ns = after.teardown_ns - before.teardown_ns,
        .admit_ns = after.admit_ns - before.admit_ns,
    };
}

static ShadowSpillPressureFitWorkDiagnostics work_delta(
    ShadowSpillPressureFitWorkDiagnostics after,
    ShadowSpillPressureFitWorkDiagnostics before
) {
    return (ShadowSpillPressureFitWorkDiagnostics){
        .residency_cache_hits =
            after.residency_cache_hits - before.residency_cache_hits,
        .residency_cache_misses =
            after.residency_cache_misses - before.residency_cache_misses,
        .schedule_emissions =
            after.schedule_emissions - before.schedule_emissions,
        .schedule_cache_hits =
            after.schedule_cache_hits - before.schedule_cache_hits,
        .simulation_calls = after.simulation_calls - before.simulation_calls,
        .simulation_cache_hits =
            after.simulation_cache_hits - before.simulation_cache_hits,
        .admission_calls = after.admission_calls - before.admission_calls,
        .sections = section_delta(after.sections, before.sections),
    };
}

static void add_repairs(
    ShadowSpillPressureFitRepairDiagnostics *destination,
    const ShadowSpillPressureFitRepairDiagnostics *source
) {
    destination->admission_prefetch_advance_attempts +=
        source->admission_prefetch_advance_attempts;
    destination->admission_prefetch_delay_attempts +=
        source->admission_prefetch_delay_attempts;
    destination->admission_pressure_boundary_attempts +=
        source->admission_pressure_boundary_attempts;
    destination->simulation_prefetch_delay_attempts +=
        source->simulation_prefetch_delay_attempts;
    destination->simulation_pressure_boundary_attempts +=
        source->simulation_pressure_boundary_attempts;
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

static void pack_boolean_cells(
    const uint8_t *source,
    size_t count,
    uint8_t *destination
) {
    size_t packed_count = count / 8U + (count % 8U != 0U);
    memset(destination, 0, packed_count);
    for (size_t index = 0U; index < count; ++index) {
        if (source[index] != 0U) {
            destination[index >> 3U] |= (uint8_t)(1U << (index & 7U));
        }
    }
}

static void unpack_boolean_cells(
    const uint8_t *source,
    size_t count,
    uint8_t *destination
) {
    static _Thread_local uint8_t expanded[256][8];
    static _Thread_local int expanded_ready = 0;
    if (!expanded_ready) {
        for (uint32_t value = 0U; value < 256U; ++value) {
            for (uint32_t bit = 0U; bit < 8U; ++bit) {
                expanded[value][bit] = (uint8_t)((value >> bit) & 1U);
            }
        }
        expanded_ready = 1;
    }
    size_t complete = count / 8U;
    for (size_t index = 0U; index < complete; ++index) {
        memcpy(destination + index * 8U, expanded[source[index]], 8U);
    }
    for (size_t index = complete * 8U; index < count; ++index) {
        destination[index] =
            (uint8_t)((source[index >> 3U] >> (index & 7U)) & 1U);
    }
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
    const ShadowSpillResidencyOptions *options
) {
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = hash_bytes(
        hash,
        &options->minimize_transfer,
        sizeof(options->minimize_transfer)
    );
    hash = hash_bytes(
        hash,
        &options->prefetch_headroom,
        sizeof(options->prefetch_headroom)
    );
    return hash_bytes(
        hash,
        workspace->extra_pressure,
        workspace->residency_cache.pressure_count *
            sizeof(*workspace->extra_pressure)
    );
}

static uint64_t schedule_cache_hash(
    uint64_t residency_content_hash,
    uint8_t rule,
    uint8_t coalesced,
    uint8_t prefetch_headroom
) {
    uint64_t hash = UINT64_C(1469598103934665603);
    hash = hash_bytes(
        hash,
        &residency_content_hash,
        sizeof(residency_content_hash)
    );
    hash = hash_bytes(hash, &rule, sizeof(rule));
    hash = hash_bytes(hash, &coalesced, sizeof(coalesced));
    return hash_bytes(hash, &prefetch_headroom, sizeof(prefetch_headroom));
}

static uint64_t indexed_schedule_hash(const ShadowSpillIndexedSchedule *schedule) {
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
    return rule <= SHADOWSPILL_PREFETCH_DEMAND;
}

static int problem_valid(
    const ShadowSpillPressureFitProblem *problem,
    const ShadowSpillPressureFitProblemOptions *options
) {
    if (problem == NULL || options == NULL || problem->residency == NULL ||
        problem->simulation == NULL || problem->seed_resident == NULL ||
        problem->seed_breaks == NULL || problem->alias_json_names == NULL ||
        problem->task_json_names == NULL ||
        problem->abi_version != SHADOWSPILL_ABI_VERSION ||
        problem->residency->abi_version != SHADOWSPILL_ABI_VERSION ||
        problem->simulation->abi_version != SHADOWSPILL_ABI_VERSION ||
        problem->simulation->task_count == 0U ||
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
    for (uint32_t alias = 0U; alias < problem->residency->alias_count; ++alias) {
        if (problem->alias_json_names[alias] == NULL) {
            return 0;
        }
    }
    for (uint32_t task = 0U; task < problem->simulation->task_count; ++task) {
        if (problem->task_json_names[task] == NULL) {
            return 0;
        }
    }
    return 1;
}

static int simulation_workspace_create(
    const ShadowSpillPressureFitProblem *problem,
    SimulationWorkspace *workspace
) {
    memset(workspace, 0, sizeof(*workspace));
    workspace->task_capacity = problem->simulation->task_count;
    workspace->device_capacity = problem->simulation->device_count;
    workspace->tasks = calloc(
        workspace->task_capacity == 0U ? 1U : workspace->task_capacity,
        sizeof(*workspace->tasks)
    );
    workspace->transfers = calloc(
        1U,
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

static int simulation_workspace_reserve_transfers(
    SimulationWorkspace *workspace,
    uint32_t capacity
) {
    if (capacity <= workspace->transfer_capacity) {
        return 0;
    }
    uint32_t selected = workspace->transfer_capacity == 0U
        ? 64U
        : workspace->transfer_capacity;
    while (selected < capacity) {
        if (selected > UINT32_MAX / 2U) {
            selected = capacity;
            break;
        }
        selected *= 2U;
    }
    ShadowSpillTransferInterval *replacement = realloc(
        workspace->transfers,
        (size_t)selected * sizeof(*replacement)
    );
    if (replacement == NULL) {
        return -1;
    }
    workspace->transfers = replacement;
    workspace->transfer_capacity = selected;
    return 0;
}

static void simulation_workspace_destroy(SimulationWorkspace *workspace) {
    free(workspace->tasks);
    free(workspace->transfers);
    free(workspace->peaks);
    memset(workspace, 0, sizeof(*workspace));
}

static int simulate_schedule(
    const ShadowSpillPressureFitProblem *problem,
    const ShadowSpillIndexedSchedule *schedule,
    SimulationWorkspace *workspace,
    ShadowSpillCandidateAdmissionWorkspace *admission_workspace,
    ShadowSpillCapacityViolation *first_violation,
    ShadowSpillSimulationResult *result,
    ShadowSpillStatus *admission_status,
    ShadowSpillAdmissionReplayResult *admission_result
) {
    if (simulation_workspace_reserve_transfers(
            workspace,
            schedule->action_count
        ) != 0) {
        return -1;
    }
    ShadowSpillSimulationProgram program;
    *admission_status = SHADOWSPILL_STATUS_OK;
    memset(admission_result, 0, sizeof(*admission_result));
    if (problem->admission == NULL) {
        shadowspill_bind_indexed_schedule(problem->simulation, schedule, &program);
    } else {
        *admission_status = shadowspill_admit_indexed_schedule(
            problem,
            schedule,
            admission_workspace,
            &program,
            admission_result
        );
        if (*admission_status == SHADOWSPILL_STATUS_REPLAY_INFEASIBLE) {
            memset(result, 0, sizeof(*result));
            return 0;
        }
        if (*admission_status != SHADOWSPILL_STATUS_OK) {
            return -1;
        }
    }
    *result = (ShadowSpillSimulationResult){
        .task_intervals = workspace->tasks,
        .task_interval_capacity = workspace->task_capacity,
        .transfer_intervals = workspace->transfers,
        .transfer_interval_capacity = workspace->transfer_capacity,
        .device_peaks = workspace->peaks,
        .device_peak_capacity = workspace->device_capacity,
        /* One slot: the count still reports the true total, and repair only
         * ever aims at the first place the plan came up short. */
        .capacity_violations = first_violation,
        .capacity_violation_capacity = 1U,
    };
    (void)shadowspill_simulate(&program, result);
    return 0;
}


/* Grow the placement buffers to hold one plan's operations and leases. */
static int placement_reserve(
    PlacementWorkspace *workspace,
    uint64_t operations,
    uint64_t leases,
    uint32_t aliases,
    uint32_t allocation_slots
) {
    /* A program can legitimately have none of a kind -- no allocation steps,
     * no aliases. Reserving one anyway keeps every buffer a valid pointer,
     * which is what the builders below validate. */
    operations = operations ? operations : 1U;
    leases = leases ? leases : 1U;
    aliases = aliases ? aliases : 1U;
    allocation_slots = allocation_slots ? allocation_slots : 1U;
    if (operations > workspace->operation_capacity) {
        uint64_t want = operations;
        free(workspace->lease_ids);
        free(workspace->dependency_ids);
        free(workspace->bytes);
        free(workspace->alignments);
        free(workspace->kinds);
        free(workspace->purposes);
        free(workspace->boundaries);
        free(workspace->indices);
        free(workspace->allocation_offsets);
        workspace->lease_ids = malloc(want * sizeof(*workspace->lease_ids));
        workspace->dependency_ids =
            malloc(want * sizeof(*workspace->dependency_ids));
        workspace->bytes = malloc(want * sizeof(*workspace->bytes));
        workspace->alignments = malloc(want * sizeof(*workspace->alignments));
        workspace->kinds = malloc(want * sizeof(*workspace->kinds));
        workspace->purposes = malloc(want * sizeof(*workspace->purposes));
        workspace->boundaries = malloc(want * sizeof(*workspace->boundaries));
        workspace->indices = malloc(want * sizeof(*workspace->indices));
        workspace->allocation_offsets =
            malloc(want * sizeof(*workspace->allocation_offsets));
        workspace->operation_capacity = want;
        if (workspace->lease_ids == NULL || workspace->dependency_ids == NULL ||
            workspace->bytes == NULL || workspace->alignments == NULL ||
            workspace->kinds == NULL || workspace->purposes == NULL ||
            workspace->boundaries == NULL || workspace->indices == NULL ||
            workspace->allocation_offsets == NULL) {
            return -1;
        }
    }
    if (leases > workspace->lease_capacity) {
        free(workspace->lifetimes);
        free(workspace->identities);
        free(workspace->offsets);
        free(workspace->lease_aliases);
        free(workspace->lease_starts);
        free(workspace->lease_retires);
        workspace->lease_aliases =
            malloc(leases * sizeof(*workspace->lease_aliases));
        workspace->lease_starts =
            malloc(leases * sizeof(*workspace->lease_starts));
        workspace->lease_retires =
            malloc(leases * sizeof(*workspace->lease_retires));
        workspace->lifetimes = malloc(leases * sizeof(*workspace->lifetimes));
        workspace->identities = malloc(leases * sizeof(*workspace->identities));
        workspace->offsets = malloc(leases * sizeof(*workspace->offsets));
        workspace->lease_capacity = leases;
        if (workspace->lifetimes == NULL || workspace->identities == NULL ||
            workspace->offsets == NULL || workspace->lease_aliases == NULL ||
            workspace->lease_starts == NULL ||
            workspace->lease_retires == NULL) {
            return -1;
        }
    }
    if (allocation_slots > workspace->allocation_slot_capacity) {
        /* One entry per flattened allocation step, which is neither a lease
         * nor an alias count. */
        free(workspace->allocation_step_leases);
        workspace->allocation_step_leases = malloc(
            (size_t)allocation_slots * sizeof(*workspace->allocation_step_leases)
        );
        workspace->allocation_slot_capacity = allocation_slots;
        if (workspace->allocation_step_leases == NULL) {
            return -1;
        }
    }
    if (aliases > workspace->alias_capacity) {
        free(workspace->alias_leases);
        free(workspace->dynamic_aliases);
        workspace->alias_leases =
            malloc((size_t)aliases * sizeof(*workspace->alias_leases));
        workspace->dynamic_aliases =
            malloc((size_t)aliases * sizeof(*workspace->dynamic_aliases));
        workspace->alias_capacity = aliases;
        if (workspace->alias_leases == NULL ||
            workspace->dynamic_aliases == NULL) {
            return -1;
        }
    }
    return 0;
}

static void placement_workspace_destroy(PlacementWorkspace *workspace) {
    free(workspace->lease_ids);
    free(workspace->dependency_ids);
    free(workspace->bytes);
    free(workspace->alignments);
    free(workspace->kinds);
    free(workspace->purposes);
    free(workspace->boundaries);
    free(workspace->indices);
    free(workspace->allocation_offsets);
    free(workspace->lease_aliases);
    free(workspace->lease_starts);
    free(workspace->lease_retires);
    free(workspace->lifetimes);
    free(workspace->identities);
    free(workspace->allocation_step_leases);
    free(workspace->alias_leases);
    free(workspace->offsets);
    free(workspace->dynamic_aliases);
    memset(workspace, 0, sizeof(*workspace));
}

/*
 * Can this plan be placed in the execution pool, and if not, by how much did
 * it overrun?
 *
 * The simulator answers whether a plan fits by bytes; this answers whether it
 * fits by *placement*, which is a stricter question -- leases need contiguous
 * ranges and a pool with room in total can still have nowhere to put one.
 * A plan the pool cannot place cannot run, so the overage is what the plan
 * has to give back before it is worth anything.
 *
 * Returns 0 on success, writing `required_bytes`; -1 if the measurement could
 * not be taken at all, which is a different thing from a plan that does not
 * fit.
 */
static int place_plan(
    const ShadowSpillPressureFitProblem *problem,
    CandidateWorkspace *workspace,
    const ShadowSpillSimulationResult *simulation,
    uint64_t *required_bytes
) {
    const ShadowSpillIndexedSchedule *schedule = &workspace->schedule.value;
    const ShadowSpillAdmissionFacts *admission = problem->placement;
    if (admission == NULL) {
        return -1;
    }
    /* A result the admission replay refused is zeroed, and a zeroed status
     * reads as OK, so a plan can arrive here with no intervals behind it.
     * Lease lifetimes are derived from those intervals, so there is nothing
     * to place. */
    if (simulation->task_intervals == NULL ||
        simulation->transfer_intervals == NULL ||
        simulation->makespan_ns == 0U) {
        return -1;
    }
    uint64_t operation_capacity = 0U;
    uint64_t lease_capacity = 0U;
    ShadowSpillStatus bounds_status = shadowspill_admission_operation_bounds(
        problem->simulation, admission, schedule,
        &operation_capacity, &lease_capacity
    );
    if (bounds_status != SHADOWSPILL_STATUS_OK) {
        return -1;
    }
    PlacementWorkspace *place = &workspace->placement;
    if (placement_reserve(
            place, operation_capacity, lease_capacity,
            problem->residency->alias_count,
            /* Flattened allocation steps, which is where the topology ends --
             * not the slot count, which counts something else. */
            admission->task_allocation_offsets[admission->task_count]
        ) != 0) {
        return -1;
    }
    place->operations = (ShadowSpillAdmissionOperations){
        .lease_ids = place->lease_ids,
        .dependency_ids = place->dependency_ids,
        .bytes = place->bytes,
        .alignments = place->alignments,
        .kinds = place->kinds,
        .purposes = place->purposes,
        .boundaries = place->boundaries,
        .indices = place->indices,
        .allocation_offsets = place->allocation_offsets,
        .operation_capacity = operation_capacity,
        .lease_aliases = place->lease_aliases,
        .lease_starts = place->lease_starts,
        .lease_retires = place->lease_retires,
        .lease_capacity = lease_capacity,
    };
    ShadowSpillStatus operations_status = shadowspill_build_admission_operations(
        problem->simulation, admission, schedule, &place->operations
    );
    if (operations_status != SHADOWSPILL_STATUS_OK) {
        return -1;
    }
    /* Aliases the plan leaves resident on the device hold their lease past the
     * step, so placement has to keep room for them. */
    uint32_t dynamic_count = 0U;
    for (uint32_t index = 0U; index < schedule->final_count; ++index) {
        if (schedule->final_locations[index] == SHADOWSPILL_MEMORY_DEVICE) {
            place->dynamic_aliases[dynamic_count++] = schedule->final_aliases[index];
        }
    }
    ShadowSpillLeaseLifetimeProblem lifetime_problem = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .operations = &place->operations,
        .admission = admission,
        .schedule = schedule,
        .task_intervals = simulation->task_intervals,
        .task_interval_count = simulation->task_interval_count,
        .transfer_intervals = simulation->transfer_intervals,
        .transfer_interval_count = simulation->transfer_interval_count,
        .makespan_ns = simulation->makespan_ns,
        .dynamic_aliases = place->dynamic_aliases,
        .dynamic_alias_count = dynamic_count,
    };
    ShadowSpillLeaseLifetimeResult lifetime_result = {
        .lifetimes = place->lifetimes,
        .identities = place->identities,
        .allocation_step_leases = place->allocation_step_leases,
        .alias_leases = place->alias_leases,
    };
    ShadowSpillStatus lifetime_status =
        shadowspill_build_lease_lifetimes(&lifetime_problem, &lifetime_result);
    if (lifetime_status != SHADOWSPILL_STATUS_OK) {
        return -1;
    }
    /* Fixed leases occupy the prefix, so placement runs on it without a copy. */
    ShadowSpillPlacementProblem placement_problem = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .lifetime_count = (uint32_t)lifetime_result.fixed_count,
        .lifetimes = place->lifetimes,
    };
    ShadowSpillPlacementResult placement_result = {
        .required_bytes = 0U,
        .offsets = place->offsets,
    };
    ShadowSpillStatus place_status =
        shadowspill_place_lifetimes(&placement_problem, &placement_result);
    if (place_status != SHADOWSPILL_STATUS_OK) {
        return -1;
    }
    /* What the pool has to hold is the reusable slice plus the leases that
     * outlive the step, which are placed outside it. The certificate adds the
     * same two, so measuring only the slice reports a plan as fitting that
     * the certificate then refuses. */
    uint64_t dynamic_bytes = 0U;
    for (uint64_t lease = lifetime_result.fixed_count;
         lease < lifetime_result.lifetime_count;
         ++lease) {
        if (place->lifetimes[lease].bytes >
            UINT64_MAX - dynamic_bytes) {
            return -1;
        }
        dynamic_bytes += place->lifetimes[lease].bytes;
    }
    if (placement_result.required_bytes > UINT64_MAX - dynamic_bytes) {
        return -1;
    }
    *required_bytes = placement_result.required_bytes + dynamic_bytes;
    return 0;
}

static int candidate_workspace_create(
    const ShadowSpillPressureFitProblem *problem,
    CandidateWorkspace *workspace
) {
    memset(workspace, 0, sizeof(*workspace));
    uint64_t cell_count = (uint64_t)problem->residency->alias_count *
        problem->residency->boundary_count;
    uint64_t pressure_count = (uint64_t)problem->residency->device_count *
        problem->residency->boundary_count;
    if (cell_count > SIZE_MAX || pressure_count > SIZE_MAX) {
        return -1;
    }
    size_t cells = (size_t)cell_count;
    workspace->residency_cache.cell_count = cells;
    workspace->residency_cache.packed_cell_count =
        cells / 8U + (cells % 8U != 0U);
    workspace->residency_cache.pressure_count = (size_t)pressure_count;
    workspace->resident = calloc(cells == 0U ? 1U : cells, 1U);
    workspace->breaks = calloc(cells == 0U ? 1U : cells, 1U);
    workspace->base_resident = calloc(cells == 0U ? 1U : cells, 1U);
    workspace->base_breaks = calloc(cells == 0U ? 1U : cells, 1U);
    workspace->repair_resident = calloc(cells == 0U ? 1U : cells, 1U);
    workspace->repair_breaks = calloc(cells == 0U ? 1U : cells, 1U);
    workspace->cut_scratch_capacity = problem->residency->alias_count;
    workspace->cut_scratch = calloc(
        workspace->cut_scratch_capacity == 0U
            ? 1U
            : (size_t)workspace->cut_scratch_capacity,
        sizeof(*workspace->cut_scratch)
    );
    workspace->removable_aliases = calloc(
        problem->residency->alias_count == 0U
            ? 1U
            : problem->residency->alias_count,
        1U
    );
    workspace->extra_pressure = calloc(
        pressure_count == 0U ? 1U : (size_t)pressure_count,
        sizeof(*workspace->extra_pressure)
    );
    if (workspace->resident == NULL || workspace->breaks == NULL ||
        workspace->base_resident == NULL || workspace->base_breaks == NULL ||
        workspace->repair_resident == NULL ||
        workspace->repair_breaks == NULL ||
        workspace->cut_scratch == NULL ||
        workspace->removable_aliases == NULL ||
        workspace->extra_pressure == NULL ||
        shadowspill_schedule_storage_create(
            problem->residency->alias_count,
            &workspace->schedule
        ) != 0 ||
        shadowspill_schedule_storage_create(
            problem->residency->alias_count,
            &workspace->selected
        ) != 0 ||
        shadowspill_schedule_storage_create(
            problem->residency->alias_count,
            &workspace->best
        ) != 0 ||
        simulation_workspace_create(
            problem,
            &workspace->simulation
        ) != 0 ||
        (problem->admission != NULL &&
         shadowspill_candidate_admission_workspace_create(
             problem, &workspace->admission
         ) != 0) ||
        shadowspill_residency_workspace_create(
            problem->residency,
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
    free(workspace->repair_resident);
    free(workspace->repair_breaks);
    free(workspace->removable_aliases);
    free(workspace->extra_pressure);
    free(workspace->cut_scratch);
    free(workspace->prefetch_constraints);
    shadowspill_schedule_storage_destroy(&workspace->schedule);
    shadowspill_schedule_storage_destroy(&workspace->selected);
    shadowspill_schedule_storage_destroy(&workspace->best);
    simulation_workspace_destroy(&workspace->simulation);
    placement_workspace_destroy(&workspace->placement);
    shadowspill_candidate_admission_workspace_destroy(&workspace->admission);
    shadowspill_residency_workspace_destroy(workspace->residency_workspace);
    for (uint32_t index = 0U; index < workspace->residency_cache.count; ++index) {
        free(workspace->residency_cache.entries[index].extra_pressure);
        free(workspace->residency_cache.entries[index].resident_bits);
        free(workspace->residency_cache.entries[index].break_bits);
    }
    free(workspace->residency_cache.entries);
    free(workspace->residency_cache.index.slots);
    for (uint32_t index = 0U; index < workspace->schedule_cache.count; ++index) {
        ScheduleCacheEntry *entry = &workspace->schedule_cache.entries[index];
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
        ShadowSpillIndexedSchedule *schedule =
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

static int record_prefetch_constraint(
    CandidateWorkspace *workspace,
    ShadowSpillPrefetchTriggerConstraint incoming
) {
    for (uint32_t index = 0U; index < workspace->prefetch_constraint_count;
         ++index) {
        ShadowSpillPrefetchTriggerConstraint *current =
            &workspace->prefetch_constraints[index];
        if (current->alias != incoming.alias ||
            current->consumer_task != incoming.consumer_task) {
            continue;
        }
        uint32_t minimum = current->minimum_trigger > incoming.minimum_trigger
            ? current->minimum_trigger
            : incoming.minimum_trigger;
        uint32_t maximum = current->maximum_trigger < incoming.maximum_trigger
            ? current->maximum_trigger
            : incoming.maximum_trigger;
        if (minimum > maximum) {
            return 1;
        }
        current->minimum_trigger = minimum;
        current->maximum_trigger = maximum;
        return 0;
    }
    if (workspace->prefetch_constraint_count ==
        workspace->prefetch_constraint_capacity) {
        uint32_t capacity = workspace->prefetch_constraint_capacity == 0U
            ? 8U
            : workspace->prefetch_constraint_capacity * 2U;
        if (capacity < workspace->prefetch_constraint_capacity) {
            return -1;
        }
        ShadowSpillPrefetchTriggerConstraint *constraints = realloc(
            workspace->prefetch_constraints,
            (size_t)capacity * sizeof(*constraints)
        );
        if (constraints == NULL) {
            return -1;
        }
        workspace->prefetch_constraints = constraints;
        workspace->prefetch_constraint_capacity = capacity;
    }
    workspace->prefetch_constraints[workspace->prefetch_constraint_count++] =
        incoming;
    return 0;
}


static void residency_options(
    const ShadowSpillPressureFitProblem *problem,
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
        .seed_resident = problem->seed_resident,
        .seed_breaks = problem->seed_breaks,
        .extra_pressure_bytes = workspace->extra_pressure,
    };
}

static ShadowSpillStatus reduce(
    const ShadowSpillPressureFitProblem *problem,
    CandidateWorkspace *workspace,
    const ShadowSpillResidencyOptions *options,
    uint8_t *resident,
    uint8_t *breaks,
    ShadowSpillResidencyResult *result
) {
    uint64_t cells = (uint64_t)problem->residency->alias_count *
        problem->residency->boundary_count;
    *result = (ShadowSpillResidencyResult){
        .resident = resident,
        .resident_capacity = cells,
        .breaks = breaks,
        .break_capacity = cells,
        /* Records what this reduction gives up, when anyone is recording.
         * Several reductions can run before the record is drained, so each
         * appends after the last rather than from the start. A full scratch
         * records nothing further: the reducer stops at its capacity. */
        .cut_aliases = workspace->cut_scratch == NULL
            ? NULL
            : workspace->cut_scratch + workspace->cut_scratch_count,
        .cut_capacity =
            workspace->cut_scratch_capacity - workspace->cut_scratch_count,
    };
    const ShadowSpillStatus reduced = shadowspill_reduce_residency_reusing(
        problem->residency,
        options,
        result,
        workspace->residency_workspace
    );
    workspace->cut_scratch_count += result->cut_count;
    return reduced;
}

static ResidencyCacheEntry *find_residency_cache(
    CandidateWorkspace *workspace,
    const ShadowSpillResidencyOptions *options,
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
        if (indexed.hash == hash &&
            entry->minimize_transfer == options->minimize_transfer &&
            entry->prefetch_headroom == options->prefetch_headroom &&
            memcmp(entry->extra_pressure, workspace->extra_pressure, bytes) == 0) {
            return entry;
        }
        slot = hash_index_next(&cache->index, slot);
    }
    return NULL;
}

static ResidencyCacheEntry *append_residency_cache(
    CandidateWorkspace *workspace,
    const ShadowSpillResidencyOptions *options,
    uint64_t hash,
    ShadowSpillStatus status,
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
    entry->resident_bits = malloc(
        cache->packed_cell_count == 0U ? 1U : cache->packed_cell_count
    );
    entry->break_bits = malloc(
        cache->packed_cell_count == 0U ? 1U : cache->packed_cell_count
    );
    if (entry->extra_pressure == NULL || entry->resident_bits == NULL ||
        entry->break_bits == NULL) {
        free(entry->extra_pressure);
        free(entry->resident_bits);
        free(entry->break_bits);
        memset(entry, 0, sizeof(*entry));
        return NULL;
    }
    memcpy(
        entry->extra_pressure,
        workspace->extra_pressure,
        cache->pressure_count * sizeof(*entry->extra_pressure)
    );
    pack_boolean_cells(resident, cache->cell_count, entry->resident_bits);
    pack_boolean_cells(breaks, cache->cell_count, entry->break_bits);
    entry->content_hash = hash_bytes(
        hash_bytes(
            UINT64_C(1469598103934665603),
            entry->resident_bits,
            cache->packed_cell_count
        ),
        entry->break_bits,
        cache->packed_cell_count
    );
    entry->minimize_transfer = options->minimize_transfer;
    entry->prefetch_headroom = options->prefetch_headroom;
    entry->hash = hash;
    entry->status = status;
    entry->error_device = result->error_device;
    entry->error_boundary = result->error_boundary;
    entry->required_bytes = result->required_bytes;
    entry->capacity_bytes = result->capacity_bytes;
    if (hash_index_insert(&cache->index, hash, cache->count) != 0) {
        free(entry->extra_pressure);
        free(entry->resident_bits);
        free(entry->break_bits);
        memset(entry, 0, sizeof(*entry));
        return NULL;
    }
    ++cache->count;
    return entry;
}

static ShadowSpillStatus reduce_cached(
    const ShadowSpillPressureFitProblem *problem,
    CandidateWorkspace *workspace,
    const ShadowSpillResidencyOptions *options,
    uint8_t strategy,
    uint8_t *resident,
    uint8_t *breaks,
    ShadowSpillResidencyResult *result
) {
    ResidencyCache *cache = &workspace->residency_cache;
    (void)strategy;
    uint64_t hash = residency_cache_hash(workspace, options);
    ResidencyCacheEntry *entry = find_residency_cache(workspace, options, hash);
    int cache_hit = entry != NULL;
    if (entry == NULL) {
        ++workspace->residency_cache_misses;
        ShadowSpillResidencyResult computed;
        ShadowSpillStatus status = reduce(
            problem,
            workspace,
            options,
            resident,
            breaks,
            &computed
        );
        entry = append_residency_cache(
            workspace,
            options,
            hash,
            status,
            &computed,
            resident,
            breaks
        );
        if (entry == NULL) {
            return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        }
    } else {
        ++workspace->residency_cache_hits;
    }
    workspace->current_residency_key =
        (uint32_t)(entry - workspace->residency_cache.entries);
    if (cache_hit != 0) {
        unpack_boolean_cells(entry->resident_bits, cache->cell_count, resident);
        unpack_boolean_cells(entry->break_bits, cache->cell_count, breaks);
    }
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
    uint32_t residency_key,
    uint8_t rule,
    uint8_t coalesced,
    uint8_t prefetch_headroom,
    uint64_t hash
) {
    ScheduleCache *cache = &workspace->schedule_cache;
    const ResidencyCacheEntry *current =
        &workspace->residency_cache.entries[residency_key];
    uint32_t slot = hash_index_start(&cache->index, hash);
    while (slot != UINT32_MAX &&
           cache->index.slots[slot].entry_plus_one != 0U) {
        HashSlot indexed = cache->index.slots[slot];
        ScheduleCacheEntry *entry =
            &cache->entries[indexed.entry_plus_one - 1U];
        const ResidencyCacheEntry *cached =
            &workspace->residency_cache.entries[entry->residency_key];
        if (indexed.hash == hash && entry->rule == rule &&
            entry->coalesced == coalesced &&
            entry->prefetch_headroom == prefetch_headroom &&
            cached->content_hash == current->content_hash &&
            memcmp(cached->resident_bits, current->resident_bits,
                   workspace->residency_cache.packed_cell_count) == 0 &&
            memcmp(cached->break_bits, current->break_bits,
                   workspace->residency_cache.packed_cell_count) == 0) {
            return entry;
        }
        slot = hash_index_next(&cache->index, slot);
    }
    return NULL;
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

static int indexed_schedule_equal(
    const ShadowSpillIndexedSchedule *left,
    const ShadowSpillIndexedSchedule *right
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

static SimulationCacheEntry *append_simulation_cache(
    CandidateWorkspace *workspace,
    const ShadowSpillSimulationResult *result,
    ShadowSpillStatus admission_status,
    const ShadowSpillAdmissionReplayResult *admission,
    uint64_t hash
) {
    SimulationCache *cache = &workspace->simulation_cache;
    if (cache->count == cache->capacity) {
        uint32_t capacity = cache->capacity == 0U ? 16U : cache->capacity * 2U;
        if (capacity < cache->capacity) {
            return NULL;
        }
        SimulationCacheEntry *entries = realloc(
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
    SimulationCacheEntry *entry = &cache->entries[cache->count];
    if (clone_indexed_schedule(&workspace->schedule.value, &entry->schedule) != 0) {
        return NULL;
    }
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
        return NULL;
    }
    ++cache->count;
    return entry;
}

static int simulate_cached(
    const ShadowSpillPressureFitProblem *problem,
    CandidateWorkspace *workspace,
    ShadowSpillSimulationResult *result,
    ShadowSpillStatus *admission_status,
    ShadowSpillAdmissionReplayResult *admission_result,
    ShadowSpillAdmissionAnnotation *admission_error_annotation,
    SimulationCacheEntry **selected_entry
) {
    SimulationCache *cache = &workspace->simulation_cache;
    uint64_t hash = indexed_schedule_hash(&workspace->schedule.value);
    uint32_t slot = hash_index_start(&cache->index, hash);
    while (slot != UINT32_MAX &&
           cache->index.slots[slot].entry_plus_one != 0U) {
        HashSlot indexed = cache->index.slots[slot];
        SimulationCacheEntry *entry =
            &cache->entries[indexed.entry_plus_one - 1U];
        if (indexed.hash == hash && indexed_schedule_equal(
                &entry->schedule,
                &workspace->schedule.value
            )) {
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
    if (simulate_schedule(
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
    *selected_entry = append_simulation_cache(
        workspace,
        result,
        *admission_status,
        admission_result,
        hash
    );
    return *selected_entry == NULL ? -1 : 0;
}

static ScheduleCacheEntry *append_schedule_cache(
    CandidateWorkspace *workspace,
    uint32_t residency_key,
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
    if (clone_indexed_schedule(&workspace->schedule.value, &entry->schedule) != 0) {
        memset(entry, 0, sizeof(*entry));
        return NULL;
    }
    entry->residency_key = residency_key;
    entry->rule = rule;
    entry->hash = hash;
    entry->coalesced = coalesced;
    entry->prefetch_headroom = prefetch_headroom;
    if (hash_index_insert(&cache->index, hash, cache->count) != 0) {
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
        workspace->residency_cache.entries[
            workspace->current_residency_key
        ].content_hash,
        rule,
        coalesced,
        prefetch_headroom
    );
    ScheduleCacheEntry *entry = find_schedule_cache(
        workspace,
        workspace->current_residency_key,
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
    if (shadowspill_emit_indexed_schedule(
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
        workspace->current_residency_key,
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
    const ShadowSpillPressureFitProblem *problem,
    CandidateWorkspace *workspace,
    const ShadowSpillSimulationResult *failure
) {
    if (failure->status != SHADOWSPILL_STATUS_INITIAL_DEVICE_CAPACITY &&
        failure->status != SHADOWSPILL_STATUS_PREFETCH_DEVICE_CAPACITY &&
        failure->status != SHADOWSPILL_STATUS_TASK_DEVICE_CAPACITY) {
        return 0;
    }
    if (failure->error_device == SHADOWSPILL_SIMULATOR_NO_INDEX ||
        failure->error_device >= problem->residency->device_count) {
        return 0;
    }
    int32_t boundary = -1;
    if (failure->status == SHADOWSPILL_STATUS_TASK_DEVICE_CAPACITY) {
        if (failure->error_task == SHADOWSPILL_SIMULATOR_NO_INDEX) {
            return 0;
        }
        boundary = (int32_t)failure->error_task - 1;
    } else if (failure->status ==
               SHADOWSPILL_STATUS_PREFETCH_DEVICE_CAPACITY) {
        if (failure->error_task == SHADOWSPILL_SIMULATOR_NO_INDEX) {
            return 0;
        }
        boundary = (int32_t)failure->error_task;
    }
    uint64_t capacity = failure->error_capacity_bytes != 0U
        ? failure->error_capacity_bytes
        : shadowspill_boundary_capacity(
            problem->residency,
            failure->error_device,
            (uint32_t)(boundary + 1)
        );
    uint64_t total = failure->error_used_bytes;
    if (failure->error_requested_bytes > UINT64_MAX - total) {
        total = UINT64_MAX;
    } else {
        total += failure->error_requested_bytes;
    }
    uint64_t excess = total > capacity ? total - capacity : 1U;
    uint32_t index = (uint32_t)(boundary + 1);
    uint64_t position =
        (uint64_t)failure->error_device * problem->residency->boundary_count +
        index;
    if (workspace->extra_pressure[position] > UINT64_MAX - excess) {
        workspace->extra_pressure[position] = UINT64_MAX;
    } else {
        workspace->extra_pressure[position] += excess;
    }
    return 1;
}

static int simulation_failure_may_be_repairable(
    ShadowSpillStatus status
) {
    return status == SHADOWSPILL_STATUS_INITIAL_DEVICE_CAPACITY ||
        status == SHADOWSPILL_STATUS_PREFETCH_DEVICE_CAPACITY ||
        status == SHADOWSPILL_STATUS_TASK_DEVICE_CAPACITY;
}

static int admission_failure_boundary(
    const ShadowSpillPressureFitProblem *problem,
    const ShadowSpillIndexedSchedule *schedule,
    ShadowSpillAdmissionAnnotation annotation,
    uint32_t *task,
    uint32_t *pressure_index,
    uint32_t *alias
) {
    const uint32_t no_index = SHADOWSPILL_SIMULATOR_NO_INDEX;
    *task = no_index;
    *pressure_index = no_index;
    *alias = no_index;
    switch ((ShadowSpillAdmissionBoundaryKind)annotation.boundary) {
        case SHADOWSPILL_ADMISSION_BOUNDARY_INITIAL:
            *pressure_index = 0U;
            return 1;
        case SHADOWSPILL_ADMISSION_BOUNDARY_TASK_START:
            if (annotation.index >= problem->simulation->task_count) {
                return -1;
            }
            *task = annotation.index;
            *pressure_index = annotation.index;
            return 1;
        case SHADOWSPILL_ADMISSION_BOUNDARY_TASK_COMPLETION:
            if (annotation.index >= problem->simulation->task_count) {
                return -1;
            }
            *task = annotation.index;
            *pressure_index = annotation.index + 1U;
            return 1;
        case SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_TRIGGER:
        case SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_COMPLETION:
            if (annotation.index >= schedule->action_count) {
                return -1;
            }
            *task = schedule->action_trigger_tasks[annotation.index];
            *alias = schedule->action_aliases[annotation.index];
            if (*task >= problem->simulation->task_count) {
                return -1;
            }
            *pressure_index = *task + 1U;
            return 1;
        default:
            return -1;
    }
}

static int delay_admission_prefetch(
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillAdmissionReplayResult *failure,
    ShadowSpillAdmissionAnnotation annotation,
    ShadowSpillScheduleStorage *schedule,
    ShadowSpillPrefetchTriggerConstraint *constraint
) {
    ShadowSpillSimulationResult projected = {
        .error_alias = SHADOWSPILL_SIMULATOR_NO_INDEX,
        .error_device = 0U,
        .error_capacity_bytes = facts->problem->admission->pool_capacity_bytes,
        .error_used_bytes =
            facts->problem->admission->pool_capacity_bytes -
            failure->error_free_bytes,
        .error_requested_bytes = failure->error_requested_bytes,
    };
    if (annotation.boundary == SHADOWSPILL_ADMISSION_BOUNDARY_TASK_START) {
        if (annotation.index >= facts->task_count) {
            return -1;
        }
        projected.status = SHADOWSPILL_STATUS_TASK_DEVICE_CAPACITY;
        projected.error_task = annotation.index;
    } else if (
        annotation.boundary == SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_TRIGGER &&
        annotation.index < schedule->value.action_count &&
        schedule->value.action_kinds[annotation.index] ==
            SHADOWSPILL_MEMORY_PREFETCH
    ) {
        projected.status = SHADOWSPILL_STATUS_PREFETCH_DEVICE_CAPACITY;
        projected.error_task =
            schedule->value.action_trigger_tasks[annotation.index];
        projected.error_alias =
            schedule->value.action_aliases[annotation.index];
    } else {
        return 0;
    }
    return shadowspill_delay_indexed_prefetch(
        facts, &projected, schedule, constraint
    );
}

static int advance_admission_prefetch(
    const ShadowSpillScheduleFacts *facts,
    ShadowSpillAdmissionAnnotation annotation,
    ShadowSpillScheduleStorage *schedule,
    ShadowSpillPrefetchTriggerConstraint *constraint
) {
    if (annotation.boundary !=
            SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_TRIGGER ||
        annotation.index >= schedule->value.action_count ||
        schedule->value.action_kinds[annotation.index] !=
            SHADOWSPILL_MEMORY_PREFETCH) {
        return 0;
    }
    return shadowspill_advance_indexed_prefetch_to_release(
        facts, annotation.index, schedule, constraint
    );
}

static int add_admission_repair_pressure(
    const ShadowSpillPressureFitProblem *problem,
    CandidateWorkspace *workspace,
    const ShadowSpillResidencyOptions *options,
    const ShadowSpillAdmissionReplayResult *failure,
    ShadowSpillAdmissionAnnotation annotation,
    const ShadowSpillIndexedSchedule *schedule
) {
    uint32_t task = SHADOWSPILL_SIMULATOR_NO_INDEX;
    uint32_t pressure_index = SHADOWSPILL_SIMULATOR_NO_INDEX;
    uint32_t alias = SHADOWSPILL_SIMULATOR_NO_INDEX;
    const int boundary = admission_failure_boundary(
        problem,
        schedule,
        annotation,
        &task,
        &pressure_index,
        &alias
    );
    (void)task;
    (void)alias;
    if (boundary <= 0 ||
        pressure_index >= problem->residency->boundary_count) {
        return boundary;
    }
    uint64_t required_reduction = failure->error_requested_bytes >
            failure->error_largest_free_range_bytes
        ? failure->error_requested_bytes -
            failure->error_largest_free_range_bytes
        : 1U;
    const uint64_t position = pressure_index;
    uint64_t resident_pressure = 0U;
    if (shadowspill_residency_pressure_at(
            problem->residency,
            options,
            workspace->resident,
            workspace->breaks,
            0U,
            pressure_index,
            workspace->residency_workspace,
            &resident_pressure
        ) != 0) {
        return -1;
    }
    uint64_t current_pressure = resident_pressure;
    if (current_pressure >
        UINT64_MAX - workspace->extra_pressure[position]) {
        current_pressure = UINT64_MAX;
    } else {
        current_pressure += workspace->extra_pressure[position];
    }
    const uint64_t capacity = shadowspill_boundary_capacity(
        problem->residency,
        0U,
        pressure_index
    );
    const uint64_t unused_capacity = current_pressure < capacity
        ? capacity - current_pressure
        : 0U;
    uint64_t pressure_increment = required_reduction;
    if (pressure_increment > UINT64_MAX - unused_capacity) {
        pressure_increment = UINT64_MAX;
    } else {
        pressure_increment += unused_capacity;
    }
    if (workspace->extra_pressure[position] >
        UINT64_MAX - pressure_increment) {
        workspace->extra_pressure[position] = UINT64_MAX;
    } else {
        workspace->extra_pressure[position] += pressure_increment;
    }
    return 1;
}


static void copy_admission_error(
    const ShadowSpillPressureFitProblem *problem,
    const ShadowSpillIndexedSchedule *schedule,
    const ShadowSpillAdmissionReplayResult *failure,
    ShadowSpillAdmissionAnnotation annotation,
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic
) {
    uint32_t task = SHADOWSPILL_SIMULATOR_NO_INDEX;
    uint32_t pressure_index = SHADOWSPILL_SIMULATOR_NO_INDEX;
    uint32_t alias = SHADOWSPILL_SIMULATOR_NO_INDEX;
    (void)admission_failure_boundary(
        problem,
        schedule,
        annotation,
        &task,
        &pressure_index,
        &alias
    );
    diagnostic->status = SHADOWSPILL_CANDIDATE_ADMISSION_INFEASIBLE;
    diagnostic->error_task = task;
    diagnostic->error_alias = alias;
    diagnostic->error_device = 0U;
    diagnostic->error_boundary = pressure_index ==
            SHADOWSPILL_SIMULATOR_NO_INDEX
        ? INT32_MIN
        : (int32_t)pressure_index - 1;
    diagnostic->error_capacity_bytes = problem->admission->pool_capacity_bytes;
    diagnostic->error_used_bytes =
        problem->admission->pool_capacity_bytes - failure->error_free_bytes;
    diagnostic->error_requested_bytes = failure->error_requested_bytes;
    diagnostic->error_required_bytes = failure->error_requested_bytes >
            failure->error_largest_free_range_bytes
        ? failure->error_requested_bytes -
            failure->error_largest_free_range_bytes
        : 0U;
}

static int reduce_repaired_candidate(
    const ShadowSpillPressureFitProblem *problem,
    CandidateWorkspace *workspace,
    const ShadowSpillResidencyOptions *options,
    uint8_t strategy,
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic
) {
    ShadowSpillResidencyResult residency;
    const ShadowSpillStatus status = reduce_cached(
        problem,
        workspace,
        options,
        strategy,
        workspace->resident,
        workspace->breaks,
        &residency
    );
    if (status == SHADOWSPILL_STATUS_ANALYTIC_INFEASIBLE) {
        copy_analytic_error(diagnostic, &residency);
        return 0;
    }
    return status == SHADOWSPILL_STATUS_OK ? 1 : -1;
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
    diagnostic->simulation_status = SHADOWSPILL_STATUS_OK;
    diagnostic->error_task = SHADOWSPILL_SIMULATOR_NO_INDEX;
    diagnostic->error_alias = SHADOWSPILL_SIMULATOR_NO_INDEX;
    diagnostic->error_device = SHADOWSPILL_SIMULATOR_NO_INDEX;
    diagnostic->error_boundary = INT32_MIN;
}

/*
 * Whether this candidate gets another reduction.
 *
 * Effort is the only thing that stops it. Abandoning a candidate whose plan
 * is already slower than one in hand used to be free, because reductions
 * only ever added transfers and stall: a candidate behind the incumbent
 * could not overtake it. That stopped being true when a plan that does not
 * fit began waiting rather than failing -- a reduction now relieves the
 * waiting as often as it adds to it, so a candidate behind the incumbent is
 * exactly the one with something to gain, and cutting it off there abandons
 * the plans most worth finding.
 *
 * A candidate that runs out reports `SHADOWSPILL_CANDIDATE_REPAIR_EXHAUSTED`,
 * which says the effort ran out -- "we stopped looking", never "there is no
 * plan".
 */
static int may_repair_again(
    const ShadowSpillPressureFitProblemOptions *candidate_options,
    const ShadowSpillPressureFitCandidateDiagnostic *diagnostic
) {
    return repair_total(&diagnostic->repairs) <
        candidate_options->max_repair_attempts;
}

/*
 * One candidate's search.
 *
 * A candidate starts from its strategy's base residency and repeats a fixed
 * cycle: emit a schedule, simulate it, name it, measure whether its layout
 * fits, then decide whether to keep looking. It leaves the cycle with an
 * answer, or without one when it runs out of ways to improve.
 *
 * `evaluate_candidate` is that cycle and nothing else. Every step of it is a
 * stage below, the state they share is `CandidateSearch`, and the timing
 * sections are opened and closed around the stage calls so that what a
 * section covers is exactly what its name says.
 */

/* What a stage tells the cycle to do next. */
typedef enum StageOutcome {
    /* Run the next stage of this round. */
    STAGE_NEXT = 0,
    /* Something changed; start a new round. */
    STAGE_REPEAT,
    /* The candidate is finished. `search->answer` is what to report. */
    STAGE_DONE,
} StageOutcome;

typedef struct CandidateSearch {
    /* Fixed for the whole search. */
    const ShadowSpillPressureFitProblem *problem;
    const ShadowSpillScheduleFacts *facts;
    const ShadowSpillPressureFitProblemOptions *options;
    CandidateWorkspace *workspace;
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic;
    ShadowSpillResidencyOptions reduce_options;
    uint8_t strategy;
    uint8_t rule;
    uint8_t coalesced;
    /* Whether there is a pool to place into at all. Without one the
     * candidate answers with its fastest plan, which is what a caller that
     * supplied no topology can be told. */
    int placing;
    uint64_t cells;
    uint64_t pressure_cells;

    /* Moves as the search runs. */
    int need_emit;
    /* Capacity is a property of the plan, so it travels with it. */
    uint64_t plan_capacity_bytes;
    /* The best plan the search set aside, held in `workspace->best`. */
    uint64_t best_makespan_ns;
    uint8_t best_digest[SHADOWSPILL_PLANNER_DIGEST_BYTES];
    /* The plan this candidate would answer with: the best it has placed,
     * which is not always the best it has simulated. */
    uint64_t placed_makespan_ns;
    uint8_t placed_digest[SHADOWSPILL_PLANNER_DIGEST_BYTES];

    /* The round in hand: what the last simulation produced. */
    ShadowSpillSimulationResult simulation;
    ShadowSpillStatus simulation_status;
    ShadowSpillStatus admission_status;
    ShadowSpillAdmissionReplayResult admission_result;
    ShadowSpillAdmissionAnnotation admission_annotation;
    SimulationCacheEntry *simulation_entry;
    int improves;

    /* What `evaluate_candidate` returns, set with STAGE_DONE. */
    int answer;
} CandidateSearch;

static void search_begin(
    CandidateSearch *search,
    const ShadowSpillPressureFitProblem *problem,
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillPressureFitProblemOptions *options,
    CandidateWorkspace *workspace,
    uint8_t strategy,
    uint8_t rule,
    uint8_t coalesced,
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic
) {
    memset(search, 0, sizeof(*search));
    search->problem = problem;
    search->facts = facts;
    search->options = options;
    search->workspace = workspace;
    search->diagnostic = diagnostic;
    search->strategy = strategy;
    search->rule = rule;
    search->coalesced = coalesced;
    search->placing = problem->placement != NULL;
    search->cells = (uint64_t)problem->residency->alias_count *
        problem->residency->boundary_count;
    search->pressure_cells = (uint64_t)problem->residency->device_count *
        problem->residency->boundary_count;
    search->need_emit = 1;
    search->plan_capacity_bytes = problem->placement == NULL
        ? 0U
        : problem->placement->object_capacity_bytes;
    residency_options(problem, workspace, strategy, &search->reduce_options);

    initialize_diagnostic(diagnostic, strategy, rule, coalesced);
    memset(
        workspace->extra_pressure,
        0,
        (size_t)search->pressure_cells * sizeof(*workspace->extra_pressure)
    );
    workspace->plan_capacity_given_back = 0U;
    workspace->cut_scratch_count = 0U;
    memcpy(workspace->resident, workspace->base_resident, (size_t)search->cells);
    memcpy(workspace->breaks, workspace->base_breaks, (size_t)search->cells);
    workspace->prefetch_constraint_count = 0U;
    workspace->current_residency_key = workspace->base_residency_key;
}

/* Report a plan as this candidate's answer. */
static void set_answer(
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic,
    uint64_t makespan_ns,
    const uint8_t *digest
) {
    diagnostic->status = SHADOWSPILL_CANDIDATE_VALID;
    diagnostic->makespan_ns = makespan_ns;
    memcpy(diagnostic->schedule_digest, digest, SHADOWSPILL_PLANNER_DIGEST_BYTES);
}

/* Answer with a plan the search set aside. The caller reads the answer from
 * the live schedule, so the kept plan has to move back into it. */
static int answer_with_kept(CandidateSearch *search, uint64_t makespan_ns) {
    set_answer(search->diagnostic, makespan_ns, search->best_digest);
    if (shadowspill_schedule_storage_copy(
            &search->workspace->schedule, &search->workspace->best
        ) != 0) {
        return -1;
    }
    return 1;
}

/* Stop without an answer of this candidate's own -- unless it placed a plan,
 * in which case it has one whatever became of it afterwards. The shared
 * record already names that plan, so this candidate has to report it:
 * otherwise the record holds a makespan whose plan nobody kept, and the
 * search would answer with a plan it cannot produce. */
static int answer_or_stop(CandidateSearch *search) {
    if (search->placing && search->placed_makespan_ns != 0U) {
        return answer_with_kept(search, search->placed_makespan_ns);
    }
    return 0;
}

static StageOutcome search_done(CandidateSearch *search, int answer) {
    search->answer = answer;
    return STAGE_DONE;
}

/* Add a step to the trajectory, when the caller asked for one. */
static int record_search_step(CandidateSearch *search) {
    if (search->options->record_reduction_steps == 0U) {
        return 0;
    }
    const uint32_t cuts_before = search->diagnostic->cut_count;
    if (drain_cuts(search->diagnostic, search->workspace) != 0) {
        return -1;
    }
    return record_step(
        search->diagnostic,
        (ShadowSpillPressureFitReductionStep){
            .makespan_ns = search->simulation.makespan_ns,
            .capacity_bytes = search->plan_capacity_bytes,
            .cut_offset = cuts_before,
            .cut_count = search->diagnostic->cut_count - cuts_before,
            .repairs = (uint32_t)repair_total(&search->diagnostic->repairs),
            .simulation_status = search->simulation.status,
            .capacity_violations = search->simulation.capacity_violation_count,
            .flags = search->simulation.status == SHADOWSPILL_STATUS_OK
                ? SHADOWSPILL_STEP_SIMULATED
                : 0U,
        }
    );
}

static void mark_search_step(
    CandidateSearch *search, uint32_t flags, uint64_t required_bytes
) {
    if (search->options->record_reduction_steps != 0U) {
        mark_last_step(search->diagnostic, flags, required_bytes);
    }
}

/*
 * Diagnostic-only reduction tracing, enabled by SHADOWSPILL_REDUCTION_TRACE
 * and never active in normal planning.
 *
 * Emitted after every simulation rather than only after a failing one,
 * because a reduction that succeeds is exactly the interesting case: whether
 * makespan falls monotonically as a candidate reduces, or rises and later
 * recovers. The resolved program is identified by the problem it was compiled
 * from, since a policy alone is shared across all five and grouping by it
 * merges them.
 */
static void trace_reduction(const CandidateSearch *search) {
    static _Thread_local int enabled = -1;
    if (enabled < 0) {
        enabled = getenv("SHADOWSPILL_REDUCTION_TRACE") != NULL;
    }
    if (!enabled) {
        return;
    }
    fprintf(
        stderr,
        "reduction-trace resolved=%llu strategy=%u rule=%u coalesced=%u "
        "step=%llu status=%d makespan=%llu shortfalls=%u actions=%u\n",
        (unsigned long long)(uintptr_t)search->problem,
        search->strategy,
        search->rule,
        search->coalesced,
        (unsigned long long)repair_total(&search->diagnostic->repairs),
        (int)search->simulation_status,
        (unsigned long long)search->simulation.makespan_ns,
        search->simulation.capacity_violation_count,
        search->workspace->schedule.value.action_count
    );
}

/*
 * Diagnostic-only repair tracing for the planning-efficiency investigation
 * (docs/internal/plans/planning_efficiency_0818, E012). Enabled by
 * SHADOWSPILL_REPAIR_TRACE; never active in normal planning.
 *
 * `makespan` and the transfer totals are what a dominance bound would be
 * built from, logged per repair so the investigation can see whether any of
 * them rises fast enough to cross an incumbent before the candidate converges
 * anyway. `makespan` is only meaningful when status is OK: the simulator
 * assigns it on the success path alone.
 */
static void trace_repair(const CandidateSearch *search) {
    static _Thread_local int enabled = -1;
    if (enabled < 0) {
        enabled = getenv("SHADOWSPILL_REPAIR_TRACE") != NULL;
    }
    if (!enabled) {
        return;
    }
    const ShadowSpillIndexedSchedule *schedule = &search->workspace->schedule.value;
    uint64_t fetch_bytes = 0U;
    uint64_t evict_bytes = 0U;
    for (uint32_t index = 0U; index < schedule->action_count; ++index) {
        const uint64_t bytes =
            search->problem->residency->alias_size_bytes[schedule->action_aliases[index]];
        if (schedule->action_kinds[index] == SHADOWSPILL_MEMORY_PREFETCH) {
            fetch_bytes += bytes;
        } else if (schedule->action_kinds[index] == SHADOWSPILL_MEMORY_OFFLOAD) {
            evict_bytes += bytes;
        }
    }
    fprintf(
        stderr,
        "repair-trace strategy=%u rule=%u coalesced=%u attempt=%llu status=%d "
        "makespan=%llu fetch_bytes=%llu evict_bytes=%llu actions=%u time=%llu "
        "used=%llu requested=%llu capacity=%llu\n",
        search->strategy,
        search->rule,
        search->coalesced,
        (unsigned long long)repair_total(&search->diagnostic->repairs),
        (int)search->simulation_status,
        (unsigned long long)search->simulation.makespan_ns,
        (unsigned long long)fetch_bytes,
        (unsigned long long)evict_bytes,
        schedule->action_count,
        (unsigned long long)search->simulation.error_time_ns,
        (unsigned long long)search->simulation.error_used_bytes,
        (unsigned long long)search->simulation.error_requested_bytes,
        (unsigned long long)search->simulation.error_capacity_bytes
    );
}

/* Turn the current residency into an ordered schedule. */
static StageOutcome search_emit(CandidateSearch *search) {
    CandidateWorkspace *workspace = search->workspace;
    if (search->rule == SHADOWSPILL_PREFETCH_INTERVAL_ENTRY &&
        shadowspill_extend_interval_entries(
            search->facts, workspace->resident, workspace->breaks
        ) != 0) {
        return search_done(search, -1);
    }
    if (emit_cached(
            search->facts,
            workspace,
            workspace->resident,
            workspace->breaks,
            search->rule,
            search->coalesced,
            search->reduce_options.prefetch_headroom
        ) != 0) {
        return search_done(search, -1);
    }
    const int constrained = shadowspill_apply_prefetch_trigger_constraints(
        search->facts,
        workspace->prefetch_constraints,
        workspace->prefetch_constraint_count,
        &workspace->schedule
    );
    if (constrained < 0) {
        return search_done(search, -1);
    }
    if (constrained > 0) {
        search->diagnostic->status = SHADOWSPILL_CANDIDATE_ADMISSION_INFEASIBLE;
        return search_done(search, answer_or_stop(search));
    }
    search->need_emit = 0;
    return STAGE_NEXT;
}

/* Replay the schedule for a makespan, admitting it into the pool on the way. */
static StageOutcome search_simulate(CandidateSearch *search) {
    search->admission_status = SHADOWSPILL_STATUS_OK;
    memset(&search->admission_result, 0, sizeof(search->admission_result));
    memset(&search->admission_annotation, 0, sizeof(search->admission_annotation));
    search->simulation_entry = NULL;
    if (simulate_cached(
            search->problem,
            search->workspace,
            &search->simulation,
            &search->admission_status,
            &search->admission_result,
            &search->admission_annotation,
            &search->simulation_entry
        ) != 0) {
        return search_done(search, -1);
    }
    search->simulation_status = (ShadowSpillStatus)search->simulation.status;
    if (record_search_step(search) != 0) {
        return search_done(search, -1);
    }
    trace_reduction(search);
    return STAGE_NEXT;
}

/* Record a trigger move and count it, or stop if it was already tried:
 * repeating a move the schedule already carries would loop. */
static StageOutcome record_admission_move(
    CandidateSearch *search,
    ShadowSpillPrefetchTriggerConstraint constraint,
    uint64_t *attempts
) {
    const int recorded = record_prefetch_constraint(search->workspace, constraint);
    if (recorded < 0) {
        return search_done(search, -1);
    }
    if (recorded > 0) {
        copy_admission_error(
            search->problem,
            &search->workspace->schedule.value,
            &search->admission_result,
            search->admission_annotation,
            search->diagnostic
        );
        return search_done(search, answer_or_stop(search));
    }
    ++*attempts;
    return STAGE_REPEAT;
}

/* Admission refused the schedule: move the prefetch that overran, or make
 * room for it and reduce again. */
static StageOutcome search_repair_admission(CandidateSearch *search) {
    if (search->admission_status != SHADOWSPILL_STATUS_REPLAY_INFEASIBLE) {
        return STAGE_NEXT;
    }
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic = search->diagnostic;
    if (may_repair_again(search->options, diagnostic)) {
        ShadowSpillPrefetchTriggerConstraint constraint = {0};
        int moved = advance_admission_prefetch(
            search->facts,
            search->admission_annotation,
            &search->workspace->schedule,
            &constraint
        );
        if (moved < 0) {
            return search_done(search, -1);
        }
        if (moved > 0) {
            return record_admission_move(
                search,
                constraint,
                &diagnostic->repairs.admission_prefetch_advance_attempts
            );
        }
        moved = delay_admission_prefetch(
            search->facts,
            &search->admission_result,
            search->admission_annotation,
            &search->workspace->schedule,
            &constraint
        );
        if (moved < 0) {
            return search_done(search, -1);
        }
        if (moved > 0) {
            return record_admission_move(
                search,
                constraint,
                &diagnostic->repairs.admission_prefetch_delay_attempts
            );
        }
        const int pressed = add_admission_repair_pressure(
            search->problem,
            search->workspace,
            &search->reduce_options,
            &search->admission_result,
            search->admission_annotation,
            &search->workspace->schedule.value
        );
        if (pressed < 0) {
            return search_done(search, -1);
        }
        if (pressed > 0) {
            ++diagnostic->repairs.admission_pressure_boundary_attempts;
            const int reduced = reduce_repaired_candidate(
                search->problem,
                search->workspace,
                &search->reduce_options,
                search->strategy,
                diagnostic
            );
            if (reduced < 0) {
                return search_done(search, -1);
            }
            if (reduced == 0) {
                return search_done(search, answer_or_stop(search));
            }
            search->need_emit = 1;
            return STAGE_REPEAT;
        }
    }
    copy_admission_error(
        search->problem,
        &search->workspace->schedule.value,
        &search->admission_result,
        search->admission_annotation,
        diagnostic
    );
    if (!may_repair_again(search->options, diagnostic)) {
        diagnostic->status = SHADOWSPILL_CANDIDATE_REPAIR_EXHAUSTED;
    }
    return search_done(search, answer_or_stop(search));
}

/* Name the schedule. Two candidates that reduce to the same plan get the same
 * name, which is how the search recognises a plan it has already measured. */
static StageOutcome search_digest(CandidateSearch *search) {
    if (search->simulation_entry->digest_valid == 0U) {
        shadowspill_schedule_digest(
            search->problem,
            &search->workspace->schedule.value,
            search->simulation_entry->digest
        );
        search->simulation_entry->digest_valid = 1U;
    }
    search->improves = search->best_makespan_ns == 0U ||
        search->simulation.makespan_ns < search->best_makespan_ns;
    return STAGE_NEXT;
}

/* Keep a plan whose layout fit, offering it to the shared record. */
static StageOutcome search_keep_placed(CandidateSearch *search) {
    ShadowSpillBestPlacedRecord record = {
        .makespan_ns = search->simulation.makespan_ns,
        .object_capacity_bytes = search->plan_capacity_bytes,
        .capacity_given_back_bytes = search->workspace->plan_capacity_given_back,
        .selection_index = search->options->selection_index,
        .residency_strategy = search->strategy,
        .prefetch_rule = search->rule,
        .coalesced = search->coalesced,
    };
    memcpy(
        record.schedule_digest,
        search->simulation_entry->digest,
        sizeof(record.schedule_digest)
    );
    /* Re-compared under the lock: another candidate may have placed something
     * better while this was being measured, so being admitted is not being
     * best. */
    (void)shadowspill_best_placed_offer(
        search->options->best_placed, &record, &search->workspace->schedule
    );
    ++search->diagnostic->placements_admitted;
    mark_search_step(search, SHADOWSPILL_STEP_PLACED, 0U);
    if (search->placed_makespan_ns != 0U &&
        search->simulation.makespan_ns >= search->placed_makespan_ns) {
        return STAGE_NEXT;
    }
    mark_search_step(search, SHADOWSPILL_STEP_BEST, 0U);
    search->placed_makespan_ns = search->simulation.makespan_ns;
    if (shadowspill_schedule_storage_copy(
            &search->workspace->best, &search->workspace->schedule
        ) != 0) {
        return search_done(search, -1);
    }
    memcpy(
        search->best_digest,
        search->simulation_entry->digest,
        sizeof(search->best_digest)
    );
    return STAGE_NEXT;
}

/*
 * The pool cannot place this plan, which is a fact about the plan and not
 * about the search: it gives back exactly what it overran and reduces again.
 * Uniform pressure is how a plan expresses a smaller capacity to the reducer.
 *
 * The extent does not fall byte for byte with the capacity -- on one measured
 * point a 1 GiB reduction moved it 2.2 GB -- so handing back the whole
 * overage can overshoot the capacity that would have fit, which a bounded
 * step avoids at the cost of more rounds.
 */
static StageOutcome search_refine_capacity(
    CandidateSearch *search, uint64_t required_bytes, uint64_t pool_bytes
) {
    CandidateWorkspace *workspace = search->workspace;
    const uint64_t shortfall = required_bytes - pool_bytes;
    const uint64_t step = search->options->capacity_refinement_bytes;
    const uint64_t overage = (step == 0U || shortfall < step) ? shortfall : step;
    ++search->diagnostic->capacity_refinements;
    search->plan_capacity_bytes = search->plan_capacity_bytes > overage
        ? search->plan_capacity_bytes - overage
        : 0U;
    for (uint64_t cell = 0U; cell < search->pressure_cells; ++cell) {
        workspace->extra_pressure[cell] += overage;
    }
    workspace->plan_capacity_given_back += overage;
    mark_search_step(search, SHADOWSPILL_STEP_REFINED, 0U);
    /* Plan again at the smaller capacity rather than pressing further on what
     * this capacity produced. */
    memcpy(workspace->resident, workspace->base_resident, (size_t)search->cells);
    memcpy(workspace->breaks, workspace->base_breaks, (size_t)search->cells);
    workspace->prefetch_constraint_count = 0U;
    if (reduce_repaired_candidate(
            search->problem,
            workspace,
            &search->reduce_options,
            search->strategy,
            search->diagnostic
        ) > 0) {
        search->need_emit = 1;
        return STAGE_REPEAT;
    }
    /* Nothing left to reduce. The plan in hand is all this round produced, so
     * settle with it rather than starting another round that would rebuild
     * the same thing. */
    return STAGE_NEXT;
}

/*
 * Measure whether this plan has a layout that fits.
 *
 * Every new plan that could still win is worth measuring. The gate is what
 * keeps this affordable: a plan no better than one already placed cannot
 * become the answer, so it is never measured. Skipping plans on any other
 * ground -- waiting for a local minimum, say -- can leave a candidate that
 * never placed anything at all, and a candidate with no placed plan has no
 * answer to give.
 */
static StageOutcome search_place(CandidateSearch *search) {
    if (!shadowspill_best_placed_admits(
            search->options->best_placed, search->simulation.makespan_ns
        ) ||
        memcmp(
            search->placed_digest,
            search->simulation_entry->digest,
            sizeof(search->placed_digest)
        ) == 0) {
        return STAGE_NEXT;
    }
    memcpy(
        search->placed_digest,
        search->simulation_entry->digest,
        sizeof(search->placed_digest)
    );
    const uint64_t pool_bytes = search->problem->placement == NULL
        ? 0U
        : search->problem->placement->pool_capacity_bytes;
    ++search->diagnostic->placements_attempted;
    /*
     * The simulation cache keeps makespans, not timelines: its entries drop
     * the interval arrays, which point into the shared workspace. Placement is
     * derived from those intervals, so a cache hit has to be replayed before
     * it can be measured -- otherwise a plan that simply came from the cache
     * reads as a plan that cannot be placed.
     */
    if (search->simulation.task_intervals == NULL) {
        ShadowSpillStatus replay_status = SHADOWSPILL_STATUS_OK;
        ShadowSpillAdmissionReplayResult replay_result = {0};
        if (simulate_schedule(
                search->problem,
                &search->workspace->schedule.value,
                &search->workspace->simulation,
                &search->workspace->admission,
                &search->workspace->first_violation,
                &search->simulation,
                &replay_status,
                &replay_result
            ) != 0) {
            return search_done(search, -1);
        }
    }
    uint64_t required_bytes = 0U;
    const int placed = place_plan(
        search->problem, search->workspace, &search->simulation, &required_bytes
    );
    mark_search_step(
        search,
        placed == 0 ? SHADOWSPILL_STEP_MEASURED : 0U,
        placed == 0 ? required_bytes : 0U
    );
    if (placed != 0) {
        return STAGE_NEXT;
    }
    if (required_bytes <= pool_bytes) {
        return search_keep_placed(search);
    }
    return search_refine_capacity(search, required_bytes, pool_bytes);
}

/* Present the recorded shortfall as a simulation error, so the repair path
 * below reads a success that waited the same way it reads a failure. */
static void present_shortfall_as_error(CandidateSearch *search) {
    const ShadowSpillCapacityViolation *shortfall =
        &search->workspace->first_violation;
    search->simulation.status =
        shortfall->reason == SHADOWSPILL_CAPACITY_TASK_DEVICE
            ? (uint32_t)SHADOWSPILL_STATUS_TASK_DEVICE_CAPACITY
            : (uint32_t)SHADOWSPILL_STATUS_PREFETCH_DEVICE_CAPACITY;
    search->simulation.error_task = shortfall->task;
    search->simulation.error_alias = shortfall->alias;
    search->simulation.error_device = shortfall->device;
    search->simulation.error_location = shortfall->location;
    search->simulation.error_time_ns = shortfall->time_ns;
    search->simulation.error_capacity_bytes = shortfall->capacity_bytes;
    search->simulation.error_used_bytes = shortfall->used_bytes;
    search->simulation.error_requested_bytes = shortfall->requested_bytes;
}

/* Decide whether the plan in hand is this candidate's answer. */
static StageOutcome search_settle(CandidateSearch *search) {
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic = search->diagnostic;
    /* A plan that waits for memory is valid but unfinished: the wait is time
     * it pays, and the shortfall that caused it is what repair relieves.
     * Stopping here accepts that cost untouched. */
    const int stalling = search->simulation.capacity_violation_count > 0U;
    if (stalling &&
        may_repair_again(search->options, diagnostic)) {
        /* Continuing, so this plan has to be kept: repairing past a success
         * can make it worse before it makes it better. When placing, the
         * buffer already holds the best placed plan, which outranks a faster
         * plan that has no layout. */
        if (search->improves && !search->placing) {
            search->best_makespan_ns = search->simulation.makespan_ns;
            if (shadowspill_schedule_storage_copy(
                    &search->workspace->best, &search->workspace->schedule
                ) != 0) {
                return search_done(search, -1);
            }
            memcpy(
                search->best_digest,
                search->simulation_entry->digest,
                sizeof(search->best_digest)
            );
        }
        present_shortfall_as_error(search);
        return STAGE_NEXT;
    }
    /*
     * With a pool to place into, the candidate's answer is the best plan whose
     * layout fit -- not the fastest plan it simulated. A plan that cannot be
     * placed cannot run, so offering it as an answer only pushes the rejection
     * to a later layer that has to walk capacity down to escape it.
     */
    if (search->placing) {
        if (search->placed_makespan_ns == 0U) {
            diagnostic->status = SHADOWSPILL_CANDIDATE_UNPLACEABLE;
            return search_done(search, 0);
        }
        diagnostic->capacity_violation_count =
            search->simulation.capacity_violation_count;
        return search_done(
            search, answer_with_kept(search, search->placed_makespan_ns)
        );
    }
    diagnostic->capacity_violation_count =
        search->simulation.capacity_violation_count;
    if (search->improves) {
        /* The live schedule is already the answer, so nothing needs moving. */
        set_answer(
            diagnostic,
            search->simulation.makespan_ns,
            search->simulation_entry->digest
        );
        return search_done(search, 1);
    }
    /* An earlier repair reached a better plan; the caller reads the winner
     * from the live schedule, so put it back. */
    return search_done(search, answer_with_kept(search, search->best_makespan_ns));
}

/* The plan came up short. Move the prefetch that caused it, or make room for
 * it and reduce again; if neither is possible the candidate is finished. */
static StageOutcome search_repair(CandidateSearch *search) {
    trace_repair(search);
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic = search->diagnostic;
    if (may_repair_again(search->options, diagnostic)) {
        ShadowSpillPrefetchTriggerConstraint constraint = {0};
        const int delayed = shadowspill_delay_indexed_prefetch(
            search->facts,
            &search->simulation,
            &search->workspace->schedule,
            &constraint
        );
        if (delayed < 0) {
            return search_done(search, -1);
        }
        if (delayed > 0) {
            const int recorded =
                record_prefetch_constraint(search->workspace, constraint);
            if (recorded < 0) {
                return search_done(search, -1);
            }
            if (recorded > 0) {
                diagnostic->status = SHADOWSPILL_CANDIDATE_SIMULATION_INFEASIBLE;
                copy_simulation_error(diagnostic, &search->simulation);
                return search_done(search, answer_or_stop(search));
            }
            ++diagnostic->repairs.simulation_prefetch_delay_attempts;
            return STAGE_REPEAT;
        }
        if (add_repair_pressure(
                search->problem, search->workspace, &search->simulation
            ) != 0) {
            ++diagnostic->repairs.simulation_pressure_boundary_attempts;
            const int reduced = reduce_repaired_candidate(
                search->problem,
                search->workspace,
                &search->reduce_options,
                search->strategy,
                diagnostic
            );
            if (reduced < 0) {
                return search_done(search, -1);
            }
            if (reduced > 0) {
                search->need_emit = 1;
                return STAGE_REPEAT;
            }
            if (search->best_makespan_ns != 0U) {
                return search_done(
                    search, answer_with_kept(search, search->best_makespan_ns)
                );
            }
            return search_done(search, answer_or_stop(search));
        }
    }
    /* A candidate that reached a plan keeps it. Falling through here means the
     * last repair found nothing further to try, which says the search stopped
     * improving -- not that the plan it already has stopped working. */
    if (search->best_makespan_ns != 0U) {
        return search_done(
            search, answer_with_kept(search, search->best_makespan_ns)
        );
    }
    diagnostic->status =
        !may_repair_again(search->options, diagnostic) &&
            simulation_failure_may_be_repairable(search->simulation_status)
        ? (uint32_t)SHADOWSPILL_CANDIDATE_REPAIR_EXHAUSTED
        : (uint32_t)SHADOWSPILL_CANDIDATE_SIMULATION_INFEASIBLE;
    copy_simulation_error(diagnostic, &search->simulation);
    return search_done(search, answer_or_stop(search));
}

static int evaluate_candidate(
    const ShadowSpillPressureFitProblem *problem,
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillPressureFitProblemOptions *candidate_options,
    CandidateWorkspace *workspace,
    uint8_t strategy,
    uint8_t rule,
    uint8_t coalesced,
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic
) {
    CandidateSearch search;
    search_begin(
        &search,
        problem,
        facts,
        candidate_options,
        workspace,
        strategy,
        rule,
        coalesced,
        diagnostic
    );

    while (1) {
        StageOutcome outcome = STAGE_NEXT;
        if (search.need_emit) {
            const Section emit = section_open(&workspace->sections.emit_ns);
            outcome = search_emit(&search);
            section_close(emit);
        }
        if (outcome == STAGE_NEXT) {
            const uint64_t admitted_ns = workspace->admission.time_ns;
            const Section simulate = section_open(&workspace->sections.simulate_ns);
            outcome = search_simulate(&search);
            section_close(simulate);
            /* Admission runs as part of simulating, so its time is nested
             * inside the section just closed rather than beside it. */
            workspace->sections.admit_ns +=
                workspace->admission.time_ns - admitted_ns;
        }
        if (outcome == STAGE_NEXT) {
            const Section repair = section_open(&workspace->sections.repair_ns);
            outcome = search_repair_admission(&search);
            section_close(repair);
        }
        /* A plan that simulated is a plan that could be the answer: name it,
         * measure whether it fits, and decide whether to keep looking. */
        if (outcome == STAGE_NEXT &&
            search.simulation_status == SHADOWSPILL_STATUS_OK) {
            const Section digest = section_open(&workspace->sections.digest_ns);
            outcome = search_digest(&search);
            section_close(digest);
            if (outcome == STAGE_NEXT) {
                const Section place = section_open(&workspace->sections.place_ns);
                outcome = search_place(&search);
                section_close(place);
            }
            if (outcome == STAGE_NEXT) {
                const Section settle = section_open(&workspace->sections.select_ns);
                outcome = search_settle(&search);
                section_close(settle);
            }
        }
        if (outcome == STAGE_NEXT) {
            const Section repair = section_open(&workspace->sections.repair_ns);
            outcome = search_repair(&search);
            section_close(repair);
        }
        if (outcome == STAGE_DONE) {
            return search.answer;
        }
    }
}

static int adopt_selected_schedule(
    ShadowSpillPressureFitProblemResult *result,
    CandidateWorkspace *workspace
) {
    ShadowSpillIndexedSchedule *source = &workspace->selected.value;
    ShadowSpillIndexedSchedule *destination = &result->selected_schedule;
    *destination = *source;
    memset(source, 0, sizeof(*source));
    workspace->selected.action_capacity = 0U;
    workspace->selected.initial_capacity = 0U;
    workspace->selected.final_capacity = 0U;
    return 0;
}

uint64_t shadowspill_planner_struct_size(uint32_t which) {
    switch (which) {
    case SHADOWSPILL_STRUCT_PROBLEM_OPTIONS:
        return sizeof(ShadowSpillPressureFitProblemOptions);
    case SHADOWSPILL_STRUCT_WORK_DIAGNOSTICS:
        return sizeof(ShadowSpillPressureFitWorkDiagnostics);
    case SHADOWSPILL_STRUCT_CANDIDATE_DIAGNOSTIC:
        return sizeof(ShadowSpillPressureFitCandidateDiagnostic);
    case SHADOWSPILL_STRUCT_SECTION_TIMING:
        return sizeof(ShadowSpillPressureFitSectionTiming);
    case SHADOWSPILL_STRUCT_REDUCTION_STEP:
        return sizeof(ShadowSpillPressureFitReductionStep);
    case SHADOWSPILL_STRUCT_ADMISSION_FACTS:
        return sizeof(ShadowSpillAdmissionFacts);
    case SHADOWSPILL_STRUCT_BEST_PLACED_RECORD:
        return sizeof(ShadowSpillBestPlacedRecord);
    case SHADOWSPILL_STRUCT_RESIDENCY_PROBLEM:
        return sizeof(ShadowSpillResidencyProblem);
    case SHADOWSPILL_STRUCT_RESIDENCY_RESULT:
        return sizeof(ShadowSpillResidencyResult);
    default:
        return 0U;
    }
}

void shadowspill_pressurefit_problem_result_destroy(
    ShadowSpillPressureFitProblemResult *result
) {
    if (result == NULL) {
        return;
    }
    for (uint32_t index = 0U; index < result->candidate_count; ++index) {
        free(result->candidates[index].steps);
        free(result->candidates[index].cut_aliases);
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

/*
 * One resolved program's evaluation.
 *
 * The problem level does what the candidate cycle cannot do for itself:
 * derive the facts every candidate shares, reduce once per strategy so the
 * candidates built on it start from the same residency, and keep the fastest
 * answer any of them gave. `shadowspill_evaluate_pressurefit_problem` is that
 * order and nothing else, with each stage below measured under the section it
 * belongs to.
 */
typedef struct ProblemSearch {
    const ShadowSpillPressureFitProblem *problem;
    const ShadowSpillPressureFitProblemOptions *options;
    ShadowSpillPressureFitProblemResult *result;
    ShadowSpillScheduleFacts facts;
    CandidateWorkspace workspace;
    uint32_t coalesced_count;
    /* Which candidate the loops are on, and so which diagnostic it writes. */
    uint32_t candidate_index;
} ProblemSearch;

/* What one strategy's base reduction produced. Every candidate built on that
 * strategy starts from it, so it is computed once and shared. */
typedef struct StrategyBase {
    ShadowSpillStatus status;
    ShadowSpillResidencyResult residency;
} StrategyBase;

/* One diagnostic per candidate policy, allocated before any candidate runs so
 * that a candidate can write its own without the array moving underneath it. */
static ShadowSpillStatus problem_allocate_candidates(ProblemSearch *search) {
    uint32_t count = 0U;
    if (multiply_u32(
            search->options->residency_strategy_count,
            search->options->prefetch_rule_count,
            &count
        ) != 0 ||
        multiply_u32(count, search->coalesced_count, &count) != 0) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    search->result->candidates =
        calloc(count, sizeof(*search->result->candidates));
    if (search->result->candidates == NULL) {
        search->result->status = SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    search->result->candidate_count = count;
    return SHADOWSPILL_STATUS_OK;
}

/* The facts every candidate shares, and the buffers they reuse. */
static int problem_setup(ProblemSearch *search) {
    if (shadowspill_schedule_facts_create(search->problem, &search->facts) != 0 ||
        candidate_workspace_create(search->problem, &search->workspace) != 0) {
        return -1;
    }
    /* The emitter measures against the capacity the plan being built kept,
     * which is the same array the reducer adds to its occupancy. */
    search->facts.extra_pressure = search->workspace.extra_pressure;
    return 0;
}

static void problem_teardown(ProblemSearch *search) {
    shadowspill_schedule_facts_destroy(&search->facts);
    candidate_workspace_destroy(&search->workspace);
}

/* Every failure inside the loops ends the same way: the candidate array stays
 * for the caller to release with the result, and everything the evaluation
 * itself held goes now. */
static ShadowSpillStatus problem_failed(ProblemSearch *search) {
    search->result->status = SHADOWSPILL_STATUS_PLANNER_INTERNAL_ERROR;
    problem_teardown(search);
    return SHADOWSPILL_STATUS_PLANNER_INTERNAL_ERROR;
}

/* Reduce once for this strategy, from a residency carrying no repair
 * pressure: what a candidate adds later is its own. */
static void problem_reduce_base(
    ProblemSearch *search, uint8_t strategy, StrategyBase *base
) {
    CandidateWorkspace *workspace = &search->workspace;
    const uint64_t pressure_cells =
        (uint64_t)search->problem->residency->device_count *
        search->problem->residency->boundary_count;
    memset(
        workspace->extra_pressure,
        0,
        (size_t)pressure_cells * sizeof(*workspace->extra_pressure)
    );
    ShadowSpillResidencyOptions reduce_options;
    residency_options(search->problem, workspace, strategy, &reduce_options);
    base->status = reduce_cached(
        search->problem,
        workspace,
        &reduce_options,
        strategy,
        workspace->base_resident,
        workspace->base_breaks,
        &base->residency
    );
    workspace->base_residency_key = workspace->current_residency_key;
}

/* The problem answers with the fastest candidate that answered, and keeps a
 * copy of its schedule: the workspace's live one belongs to whichever
 * candidate runs next. */
static int problem_consider(
    ProblemSearch *search,
    const ShadowSpillPressureFitCandidateDiagnostic *diagnostic
) {
    ShadowSpillPressureFitProblemResult *result = search->result;
    if (result->selected_candidate_index != SHADOWSPILL_PLANNER_NO_INDEX &&
        diagnostic->makespan_ns >= result->selected_makespan_ns) {
        return 0;
    }
    result->selected_candidate_index = search->candidate_index;
    result->selected_makespan_ns = diagnostic->makespan_ns;
    if (shadowspill_schedule_storage_copy(
            &search->workspace.selected, &search->workspace.schedule
        ) != 0) {
        return -1;
    }
    return 0;
}

/* Run one candidate policy and record what it did. */
static int problem_evaluate_candidate(
    ProblemSearch *search,
    uint8_t strategy,
    uint8_t rule,
    uint8_t coalesced,
    const StrategyBase *base
) {
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic =
        &search->result->candidates[search->candidate_index];
    initialize_diagnostic(diagnostic, strategy, rule, coalesced);
    if (base->status == SHADOWSPILL_STATUS_ANALYTIC_INFEASIBLE) {
        /* No legal cut relieves this strategy's pressure, which is a fact
         * about the strategy: every candidate built on it reports it. */
        copy_analytic_error(diagnostic, &base->residency);
        return 0;
    }
    if (base->status != SHADOWSPILL_STATUS_OK) {
        return -1;
    }
    const ShadowSpillPressureFitWorkDiagnostics before =
        workspace_work(&search->workspace);
    const uint64_t started = shadowspill_monotonic_ns();
    const int valid = evaluate_candidate(
        search->problem,
        &search->facts,
        search->options,
        &search->workspace,
        strategy,
        rule,
        coalesced,
        diagnostic
    );
    /* Every counter this candidate moved, and the span it moved them in. */
    diagnostic->work = work_delta(workspace_work(&search->workspace), before);
    section_close_total(&diagnostic->work.sections, started);
    if (valid < 0) {
        return -1;
    }
    return valid > 0 ? problem_consider(search, diagnostic) : 0;
}

/* Every candidate built on one strategy's base reduction. */
static int problem_evaluate_strategy(
    ProblemSearch *search, uint8_t strategy, const StrategyBase *base
) {
    for (uint32_t rule_index = 0U;
         rule_index < search->options->prefetch_rule_count;
         ++rule_index) {
        const uint8_t rule = search->options->prefetch_rules[rule_index];
        for (uint32_t coalesced = 0U; coalesced < search->coalesced_count;
             ++coalesced) {
            if (problem_evaluate_candidate(
                    search, strategy, rule, (uint8_t)coalesced, base
                ) != 0) {
                return -1;
            }
            ++search->candidate_index;
        }
    }
    return 0;
}

static ShadowSpillStatus problem_select(ProblemSearch *search) {
    if (search->result->selected_candidate_index ==
        SHADOWSPILL_PLANNER_NO_INDEX) {
        return SHADOWSPILL_STATUS_NO_FEASIBLE_CANDIDATE;
    }
    adopt_selected_schedule(search->result, &search->workspace);
    return SHADOWSPILL_STATUS_OK;
}

ShadowSpillStatus shadowspill_evaluate_pressurefit_problem(
    const ShadowSpillPressureFitProblem *problem,
    const ShadowSpillPressureFitProblemOptions *options,
    ShadowSpillPressureFitProblemResult *result
) {
    if (result == NULL) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    memset(result, 0, sizeof(*result));
    result->selected_candidate_index = SHADOWSPILL_PLANNER_NO_INDEX;
    result->status = SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    if (!problem_valid(problem, options)) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }

    const uint64_t started = shadowspill_monotonic_ns();
    ProblemSearch search = {
        .problem = problem,
        .options = options,
        .result = result,
        .coalesced_count = options->evaluate_coalesced != 0U ? 2U : 1U,
    };
    const ShadowSpillStatus allocated = problem_allocate_candidates(&search);
    if (allocated != SHADOWSPILL_STATUS_OK) {
        return allocated;
    }

    const Section setup = section_open(&search.workspace.sections.setup_ns);
    const int setup_failed = problem_setup(&search);
    section_close(setup);
    if (setup_failed) {
        problem_teardown(&search);
        shadowspill_pressurefit_problem_result_destroy(result);
        result->status = SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }

    for (uint32_t index = 0U; index < options->residency_strategy_count;
         ++index) {
        const uint8_t strategy = options->residency_strategies[index];
        StrategyBase base = {0};
        const Section reduce = section_open(&search.workspace.sections.reduce_ns);
        problem_reduce_base(&search, strategy, &base);
        section_close(reduce);
        if (problem_evaluate_strategy(&search, strategy, &base) != 0) {
            return problem_failed(&search);
        }
    }

    const Section select = section_open(&search.workspace.sections.select_ns);
    const ShadowSpillStatus status = problem_select(&search);
    section_close(select);

    result->status = status;
    result->work = workspace_work(&search.workspace);
    for (uint32_t index = 0U; index < result->candidate_count; ++index) {
        add_repairs(&result->repairs, &result->candidates[index].repairs);
    }
    const Section teardown =
        section_open(&result->work.sections.teardown_ns);
    problem_teardown(&search);
    section_close(teardown);
    section_close_total(&result->work.sections, started);
    return status;
}
