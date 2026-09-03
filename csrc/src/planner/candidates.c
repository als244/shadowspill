
#include "admission/internal.h"
#include "../common/platform.h"
#include "internal.h"
#include "candidates_internal.h"
#include "residency_internal.h"

#include <pthread.h>
#include <stdatomic.h>
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

/* A 128-bit fingerprint: of a residency (the cells it keeps) or of a
 * schedule (its actions and boundary residency).
 *
 * Two independent 64-bit hashes rather than one, because this is compared
 * instead of the bytes themselves. A single 64-bit hash would collide about
 * once in 2^32 distinct values, which a long search would reach; 128 bits
 * puts it beyond reach, and comparing sixteen bytes replaces comparing
 * megabytes. */
typedef struct Fingerprint {
    uint64_t low;
    uint64_t high;
} Fingerprint;

static int fingerprint_equal(Fingerprint left, Fingerprint right) {
    return left.low == right.low && left.high == right.high;
}


typedef struct ScheduleMemoEntry {
    uint64_t hash;
    Fingerprint residency;
    uint8_t rule;
    uint8_t coalesced;
    uint8_t fetch_headroom;
    ShadowSpillIndexedSchedule schedule;
} ScheduleMemoEntry;

/* The last few emitted schedules, by (residency, rule, coalescing, headroom).
 *
 * Re-emission hits are recency-local: a candidate that keeps its residency
 * while it adjusts placement asks for the same schedule again within a few
 * cycles. A fixed ring keeps those hits and bounds the memory by
 * construction; an evicted schedule is simply emitted again. */
#define SCHEDULE_MEMO_CAPACITY 16U
typedef struct ScheduleMemo {
    ScheduleMemoEntry entries[SCHEDULE_MEMO_CAPACITY];
    uint32_t count;
    uint32_t next;
} ScheduleMemo;

/* One simulated schedule's outcome, identified by the schedule's fingerprint. */
typedef struct SimulationMemoEntry {
    Fingerprint identity;
    ShadowSpillSimulationResult result;
    /* Held by value, because the buffer the simulator wrote it into belongs
     * to the workspace and the next candidate overwrites it. */
    ShadowSpillCapacityViolation first_violation;
    ShadowSpillAdmissionReplayResult admission;
    uint32_t admission_status;
    uint8_t digest[SHADOWSPILL_PLANNER_DIGEST_BYTES];
    uint8_t digest_valid;
} SimulationMemoEntry;

typedef struct SimulationMemo {
    SimulationMemoEntry *entries;
    uint32_t count;
    uint32_t capacity;
    HashIndex index;
} SimulationMemo;


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
    /* Per lease, whether placement leaves it out: the lease of an alias the
     * reducer may not cut takes a static home in the resident slice instead. */
    uint8_t *excluded;
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
    /* Cell geometry, and the problem's seed residency packed once per
     * workspace so every reduction starts from packed bitmaps. */
    size_t cell_count;
    size_t packed_cell_count;
    uint8_t *packed_seed_resident;
    uint8_t *packed_seed_breaks;
    ScheduleMemo schedule_memo;
    SimulationMemo simulation_memo;
    ShadowSpillResidencyWorkspace *residency_workspace;
    ShadowSpillFetchTriggerConstraint *fetch_constraints;
    uint32_t fetch_constraint_count;
    uint32_t fetch_constraint_capacity;
    /* Which residency the workspace currently holds, and the one every
     * candidate of this strategy starts from. Content, not position, so a
     * memo below can drop entries without invalidating anything. */
    Fingerprint current_residency;
    Fingerprint base_residency;
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
    return repairs->admission_fetch_advance_attempts +
        repairs->admission_fetch_delay_attempts +
        repairs->admission_pressure_boundary_attempts +
        repairs->simulation_fetch_delay_attempts +
        repairs->simulation_pressure_boundary_attempts;
}

static ShadowSpillPressureFitWorkDiagnostics workspace_work(
    const CandidateWorkspace *workspace
) {
    return (ShadowSpillPressureFitWorkDiagnostics){
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

static ShadowSpillPressureFitWorkDiagnostics add_work(
    ShadowSpillPressureFitWorkDiagnostics total,
    ShadowSpillPressureFitWorkDiagnostics part
) {
    total.schedule_emissions += part.schedule_emissions;
    total.schedule_cache_hits += part.schedule_cache_hits;
    total.simulation_calls += part.simulation_calls;
    total.simulation_cache_hits += part.simulation_cache_hits;
    total.admission_calls += part.admission_calls;
    total.sections.prepare_ns += part.sections.prepare_ns;
    total.sections.setup_ns += part.sections.setup_ns;
    total.sections.reduce_ns += part.sections.reduce_ns;
    total.sections.emit_ns += part.sections.emit_ns;
    total.sections.simulate_ns += part.sections.simulate_ns;
    total.sections.repair_ns += part.sections.repair_ns;
    total.sections.digest_ns += part.sections.digest_ns;
    total.sections.place_ns += part.sections.place_ns;
    total.sections.select_ns += part.sections.select_ns;
    total.sections.teardown_ns += part.sections.teardown_ns;
    total.sections.admit_ns += part.sections.admit_ns;
    /* The total and the residual add like every other span, which is what
     * keeps total == named + residual true of the sum. */
    total.sections.total_ns += part.sections.total_ns;
    total.sections.residual_ns += part.sections.residual_ns;
    return total;
}

static void add_repairs(
    ShadowSpillPressureFitRepairDiagnostics *destination,
    const ShadowSpillPressureFitRepairDiagnostics *source
) {
    destination->admission_fetch_advance_attempts +=
        source->admission_fetch_advance_attempts;
    destination->admission_fetch_delay_attempts +=
        source->admission_fetch_delay_attempts;
    destination->admission_pressure_boundary_attempts +=
        source->admission_pressure_boundary_attempts;
    destination->simulation_fetch_delay_attempts +=
        source->simulation_fetch_delay_attempts;
    destination->simulation_pressure_boundary_attempts +=
        source->simulation_pressure_boundary_attempts;
}

/* FNV-1a over `data`, eight bytes per step and a byte-wise tail. Two
 * multipliers give the two independent halves of a fingerprint. */
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

static uint64_t hash_bytes(uint64_t hash, const void *data, size_t size) {
    return hash_words(hash, data, size, UINT64_C(1099511628211));
}

static uint64_t hash_bytes_high(uint64_t hash, const void *data, size_t size) {
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
    hash = hash_bytes(hash, &residency, sizeof(residency));
    hash = hash_bytes(hash, &rule, sizeof(rule));
    hash = hash_bytes(hash, &coalesced, sizeof(coalesced));
    return hash_bytes(hash, &fetch_headroom, sizeof(fetch_headroom));
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
        .low = schedule_hash_with(schedule, UINT64_C(1469598103934665603), hash_bytes),
        .high = schedule_hash_with(schedule, UINT64_C(1099511628211), hash_bytes_high),
    };
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
    return rule <= SHADOWSPILL_FETCH_DEMAND;
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
        options->fetch_rules == NULL || options->fetch_rule_count == 0U ||
        options->coalescing_modes == NULL ||
        options->coalescing_mode_count == 0U) {
        return 0;
    }
    for (uint32_t index = 0U; index < options->coalescing_mode_count; ++index) {
        if (options->coalescing_modes[index] > 1U) {
            return 0;
        }
    }
    for (uint32_t index = 0U; index < options->residency_strategy_count;
         ++index) {
        if (!strategy_valid(options->residency_strategies[index])) {
            return 0;
        }
    }
    for (uint32_t index = 0U; index < options->fetch_rule_count; ++index) {
        if (!rule_valid(options->fetch_rules[index])) {
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
        free(workspace->excluded);
        free(workspace->lease_aliases);
        free(workspace->lease_starts);
        free(workspace->lease_retires);
        workspace->excluded = malloc(leases * sizeof(*workspace->excluded));
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
            workspace->offsets == NULL || workspace->excluded == NULL ||
            workspace->lease_aliases == NULL ||
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
    free(workspace->excluded);
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
    /* Fixed leases occupy the prefix, so placement runs on it without a copy.
     * The leases of aliases the reducer may not cut are left out: they take
     * static homes in the resident slice, laid out after the rest below. */
    const uint8_t *eligible = problem->residency->alias_evict_eligible;
    for (uint64_t lease = 0U; lease < lifetime_result.fixed_count; ++lease) {
        const uint32_t alias = place->identities[lease].alias;
        place->excluded[lease] = eligible != NULL &&
            alias != SHADOWSPILL_PLANNER_NO_INDEX && eligible[alias] == 0U;
    }
    ShadowSpillPlacementProblem placement_problem = {
        .abi_version = SHADOWSPILL_ABI_VERSION,
        .lifetime_count = (uint32_t)lifetime_result.fixed_count,
        .lifetimes = place->lifetimes,
        .excluded = place->excluded,
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
    /* The resident slice follows the main assignment: each lease left out
     * takes the next aligned home, in lease order, and the fixed range ends
     * past the last of them. */
    uint64_t extent = placement_result.required_bytes;
    for (uint64_t lease = 0U; lease < lifetime_result.fixed_count; ++lease) {
        if (place->excluded[lease] != 0U &&
            shadowspill_resident_home(
                &extent,
                place->lifetimes[lease].bytes,
                place->lifetimes[lease].alignment,
                &place->offsets[lease]
            ) != 0) {
            return -1;
        }
    }
    /* What the pool has to hold is the fixed range plus the leases that
     * outlive the step, which are placed outside it. The certificate adds the
     * same two, so measuring only the range reports a plan as fitting that
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
    if (extent > UINT64_MAX - dynamic_bytes) {
        return -1;
    }
    *required_bytes = extent + dynamic_bytes;
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
    workspace->cell_count = cells;
    workspace->packed_cell_count = shadowspill_packed_cells(cells);
    const size_t packed = workspace->packed_cell_count == 0U
        ? 1U
        : workspace->packed_cell_count;
    workspace->resident = calloc(packed, 1U);
    workspace->breaks = calloc(packed, 1U);
    workspace->base_resident = calloc(packed, 1U);
    workspace->base_breaks = calloc(packed, 1U);
    workspace->repair_resident = calloc(packed, 1U);
    workspace->repair_breaks = calloc(packed, 1U);
    workspace->packed_seed_resident = calloc(packed, 1U);
    workspace->packed_seed_breaks = calloc(packed, 1U);
    if (workspace->packed_seed_resident != NULL &&
        workspace->packed_seed_breaks != NULL) {
        for (uint64_t index = 0U; index < cells; ++index) {
            shadowspill_cell_set(
                workspace->packed_seed_resident,
                index,
                problem->seed_resident[index] != 0U
            );
            shadowspill_cell_set(
                workspace->packed_seed_breaks,
                index,
                problem->seed_breaks[index] != 0U
            );
        }
        shadowspill_canonicalize_breaks(
            workspace->packed_seed_breaks,
            workspace->packed_seed_resident,
            problem->residency->alias_count,
            problem->residency->boundary_count
        );
    }
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
        workspace->packed_seed_resident == NULL ||
        workspace->packed_seed_breaks == NULL ||
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

static void free_indexed_schedule(ShadowSpillIndexedSchedule *schedule);

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
    free(workspace->fetch_constraints);
    shadowspill_schedule_storage_destroy(&workspace->schedule);
    shadowspill_schedule_storage_destroy(&workspace->selected);
    shadowspill_schedule_storage_destroy(&workspace->best);
    simulation_workspace_destroy(&workspace->simulation);
    placement_workspace_destroy(&workspace->placement);
    shadowspill_candidate_admission_workspace_destroy(&workspace->admission);
    shadowspill_residency_workspace_destroy(workspace->residency_workspace);
    free(workspace->packed_seed_resident);
    free(workspace->packed_seed_breaks);
    for (uint32_t index = 0U; index < workspace->schedule_memo.count; ++index) {
        free_indexed_schedule(&workspace->schedule_memo.entries[index].schedule);
    }
    free(workspace->simulation_memo.entries);
    free(workspace->simulation_memo.index.slots);
    memset(workspace, 0, sizeof(*workspace));
}

static int record_fetch_constraint(
    CandidateWorkspace *workspace,
    ShadowSpillFetchTriggerConstraint incoming
) {
    for (uint32_t index = 0U; index < workspace->fetch_constraint_count;
         ++index) {
        ShadowSpillFetchTriggerConstraint *current =
            &workspace->fetch_constraints[index];
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
    if (workspace->fetch_constraint_count ==
        workspace->fetch_constraint_capacity) {
        uint32_t capacity = workspace->fetch_constraint_capacity == 0U
            ? 8U
            : workspace->fetch_constraint_capacity * 2U;
        if (capacity < workspace->fetch_constraint_capacity) {
            return -1;
        }
        ShadowSpillFetchTriggerConstraint *constraints = realloc(
            workspace->fetch_constraints,
            (size_t)capacity * sizeof(*constraints)
        );
        if (constraints == NULL) {
            return -1;
        }
        workspace->fetch_constraints = constraints;
        workspace->fetch_constraint_capacity = capacity;
    }
    workspace->fetch_constraints[workspace->fetch_constraint_count++] =
        incoming;
    return 0;
}


static void residency_options(
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
        .fetch_headroom =
            strategy == SHADOWSPILL_RESIDENCY_HEADROOM_STALL ||
                strategy == SHADOWSPILL_RESIDENCY_HEADROOM_TRANSFER
            ? 1U
            : 0U,
        .seed_resident = workspace->packed_seed_resident,
        .seed_breaks = workspace->packed_seed_breaks,
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

/* Reduce a residency and name it.
 *
 * Reductions are not cached. A repair trajectory cuts aliases monotonically,
 * so it never revisits a residency, and the only reuse a cache ever found
 * was a strategy's base residency across its rule variants -- half a percent
 * of reductions when measured, against two packed bitmaps retained per
 * visited residency. The fingerprint is what other stages key on. */
static ShadowSpillStatus reduce_residency(
    const ShadowSpillPressureFitProblem *problem,
    CandidateWorkspace *workspace,
    const ShadowSpillResidencyOptions *options,
    uint8_t strategy,
    uint8_t *resident,
    uint8_t *breaks,
    ShadowSpillResidencyResult *result
) {
    (void)strategy;
    ShadowSpillResidencyResult computed;
    ShadowSpillStatus status = reduce(
        problem,
        workspace,
        options,
        resident,
        breaks,
        &computed
    );
    workspace->current_residency = (Fingerprint){
        .low = hash_bytes(
            hash_bytes(
                UINT64_C(1469598103934665603),
                resident,
                workspace->packed_cell_count
            ),
            breaks,
            workspace->packed_cell_count
        ),
        .high = hash_bytes_high(
            hash_bytes_high(
                UINT64_C(1099511628211),
                resident,
                workspace->packed_cell_count
            ),
            breaks,
            workspace->packed_cell_count
        ),
    };
    *result = (ShadowSpillResidencyResult){
        .status = (uint32_t)status,
        .error_device = computed.error_device,
        .error_boundary = computed.error_boundary,
        .required_bytes = computed.required_bytes,
        .capacity_bytes = computed.capacity_bytes,
        .resident = resident,
        .resident_capacity = workspace->cell_count,
        .breaks = breaks,
        .break_capacity = workspace->cell_count,
    };
    return status;
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
            fingerprint_equal(entry->residency, residency)) {
            return entry;
        }
    }
    return NULL;
}

static void free_indexed_schedule(ShadowSpillIndexedSchedule *schedule) {
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

static int simulate_cached(
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
            fingerprint_equal(entry->identity, identity)) {
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
        free_indexed_schedule(&entry->schedule);
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

static int emit_cached(
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
        failure->status != SHADOWSPILL_STATUS_FETCH_DEVICE_CAPACITY &&
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
               SHADOWSPILL_STATUS_FETCH_DEVICE_CAPACITY) {
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
        status == SHADOWSPILL_STATUS_FETCH_DEVICE_CAPACITY ||
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

static int delay_admission_fetch(
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillAdmissionReplayResult *failure,
    ShadowSpillAdmissionAnnotation annotation,
    ShadowSpillScheduleStorage *schedule,
    ShadowSpillFetchTriggerConstraint *constraint
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
            SHADOWSPILL_MEMORY_FETCH
    ) {
        projected.status = SHADOWSPILL_STATUS_FETCH_DEVICE_CAPACITY;
        projected.error_task =
            schedule->value.action_trigger_tasks[annotation.index];
        projected.error_alias =
            schedule->value.action_aliases[annotation.index];
    } else {
        return 0;
    }
    return shadowspill_delay_indexed_fetch(
        facts, &projected, schedule, constraint
    );
}

static int advance_admission_fetch(
    const ShadowSpillScheduleFacts *facts,
    ShadowSpillAdmissionAnnotation annotation,
    ShadowSpillScheduleStorage *schedule,
    ShadowSpillFetchTriggerConstraint *constraint
) {
    if (annotation.boundary !=
            SHADOWSPILL_ADMISSION_BOUNDARY_ACTION_TRIGGER ||
        annotation.index >= schedule->value.action_count ||
        schedule->value.action_kinds[annotation.index] !=
            SHADOWSPILL_MEMORY_FETCH) {
        return 0;
    }
    return shadowspill_advance_indexed_fetch_to_release(
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
    const ShadowSpillStatus status = reduce_residency(
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
    diagnostic->repairs_at_best = UINT32_MAX;
    diagnostic->status = SHADOWSPILL_CANDIDATE_INTERNAL_ERROR;
    diagnostic->residency_strategy = strategy;
    diagnostic->fetch_rule = rule;
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
    /* The problem's facts, held by value so this candidate can point them at
     * the capacity *it* has given back. The problem shares one set of facts
     * between workers, but what a plan gave back belongs to the worker
     * building it, so the pointer cannot live in the shared copy. */
    ShadowSpillScheduleFacts facts;
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
    /* The plan last placed, by fingerprint: placing it again is skipped. */
    Fingerprint placed_identity;

    /* The round in hand: what the last simulation produced. */
    ShadowSpillSimulationResult simulation;
    ShadowSpillStatus simulation_status;
    ShadowSpillStatus admission_status;
    ShadowSpillAdmissionReplayResult admission_result;
    ShadowSpillAdmissionAnnotation admission_annotation;
    SimulationMemoEntry *simulation_entry;
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
    /* The emitter measures against the capacity the plan being built kept,
     * which is the same array the reducer adds to its occupancy. Without
     * this the emitter packs against a capacity the plan no longer has, and
     * refining capacity never shrinks the layout it produces. */
    search->facts = *facts;
    search->facts.extra_pressure = workspace->extra_pressure;
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
    residency_options(workspace, strategy, &search->reduce_options);

    initialize_diagnostic(diagnostic, strategy, rule, coalesced);
    memset(
        workspace->extra_pressure,
        0,
        (size_t)search->pressure_cells * sizeof(*workspace->extra_pressure)
    );
    workspace->plan_capacity_given_back = 0U;
    workspace->cut_scratch_count = 0U;
    memcpy(workspace->resident, workspace->base_resident, workspace->packed_cell_count);
    memcpy(workspace->breaks, workspace->base_breaks, workspace->packed_cell_count);
    workspace->fetch_constraint_count = 0U;
    workspace->current_residency = workspace->base_residency;
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
        if (schedule->action_kinds[index] == SHADOWSPILL_MEMORY_FETCH) {
            fetch_bytes += bytes;
        } else if (schedule->action_kinds[index] == SHADOWSPILL_MEMORY_EVICT) {
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
    if (search->rule == SHADOWSPILL_FETCH_INTERVAL_ENTRY &&
        shadowspill_extend_interval_entries(
            &search->facts, workspace->resident, workspace->breaks
        ) != 0) {
        return search_done(search, -1);
    }
    if (emit_cached(
            &search->facts,
            workspace,
            workspace->resident,
            workspace->breaks,
            search->rule,
            search->coalesced,
            search->reduce_options.fetch_headroom
        ) != 0) {
        return search_done(search, -1);
    }
    const int constrained = shadowspill_apply_fetch_trigger_constraints(
        &search->facts,
        workspace->fetch_constraints,
        workspace->fetch_constraint_count,
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
    ShadowSpillFetchTriggerConstraint constraint,
    uint64_t *attempts
) {
    const int recorded = record_fetch_constraint(search->workspace, constraint);
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

/* Admission refused the schedule: move the fetch that overran, or make
 * room for it and reduce again. */
static StageOutcome search_repair_admission(CandidateSearch *search) {
    if (search->admission_status != SHADOWSPILL_STATUS_REPLAY_INFEASIBLE) {
        return STAGE_NEXT;
    }
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic = search->diagnostic;
    if (may_repair_again(search->options, diagnostic)) {
        ShadowSpillFetchTriggerConstraint constraint = {0};
        int moved = advance_admission_fetch(
            &search->facts,
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
                &diagnostic->repairs.admission_fetch_advance_attempts
            );
        }
        moved = delay_admission_fetch(
            &search->facts,
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
                &diagnostic->repairs.admission_fetch_delay_attempts
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
/* The schedule's name, computed the first time a stage needs it: a plan that
 * neither improves on the candidate's best nor reaches placement is never
 * named, and most plans are such. */
static const uint8_t *schedule_name(CandidateSearch *search) {
    if (search->simulation_entry->digest_valid == 0U) {
        shadowspill_schedule_digest(
            search->problem,
            &search->workspace->schedule.value,
            search->simulation_entry->digest
        );
        search->simulation_entry->digest_valid = 1U;
    }
    return search->simulation_entry->digest;
}

static StageOutcome search_digest(CandidateSearch *search) {
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
        .residency_strategy = search->strategy,
        .fetch_rule = search->rule,
        .coalesced = search->coalesced,
    };
    memcpy(
        record.schedule_digest,
        schedule_name(search),
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
    search->diagnostic->repairs_at_best =
        (uint32_t)repair_total(&search->diagnostic->repairs);
    search->placed_makespan_ns = search->simulation.makespan_ns;
    if (shadowspill_schedule_storage_copy(
            &search->workspace->best, &search->workspace->schedule
        ) != 0) {
        return search_done(search, -1);
    }
    memcpy(search->best_digest, schedule_name(search), sizeof(search->best_digest));
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
    memcpy(workspace->resident, workspace->base_resident, workspace->packed_cell_count);
    memcpy(workspace->breaks, workspace->base_breaks, workspace->packed_cell_count);
    workspace->fetch_constraint_count = 0U;
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

/* A plan no better than what this candidate already placed cannot become
 * its answer, whatever the mode. */
static int candidate_refuses(const CandidateSearch *search) {
    return search->placed_makespan_ns != 0U &&
        search->simulation.makespan_ns >= search->placed_makespan_ns;
}

/* Whether the shared best-placed record refuses this makespan. Lock-free
 * and possibly stale, which costs at most a measurement that would have
 * been skipped. */
static int shared_refuses(const CandidateSearch *search) {
    const uint64_t bound =
        shadowspill_best_placed_bound(search->options->best_placed);
    return bound != 0U && search->simulation.makespan_ns >= bound;
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
 *
 * A skipped measurement is not neutral: it also skips the capacity
 * refinement a failed placement would have triggered, so a skip edits the
 * candidate's whole descent. The default consults the shared record, which
 * makes a candidate's outcome depend on what other workers had placed by
 * then -- near-tied points settle on different plans run to run. That
 * variance is an accepted trade for the wall time the gate saves.
 * Deterministic mode declines the trade: its gate is candidate-local, so
 * every outcome is a pure function of its inputs and parallel evaluation
 * reproduces exactly, at the cost of measuring plans the shared bound
 * would have skipped.
 */
static StageOutcome search_place(CandidateSearch *search) {
    const int refused = search->options->deterministic
        ? candidate_refuses(search)
        : shared_refuses(search);
    if (refused ||
        fingerprint_equal(
            search->placed_identity, search->simulation_entry->identity
        )) {
        return STAGE_NEXT;
    }
    search->placed_identity = search->simulation_entry->identity;
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
            : (uint32_t)SHADOWSPILL_STATUS_FETCH_DEVICE_CAPACITY;
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
                schedule_name(search),
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
            schedule_name(search)
        );
        return search_done(search, 1);
    }
    /* An earlier repair reached a better plan; the caller reads the winner
     * from the live schedule, so put it back. */
    return search_done(search, answer_with_kept(search, search->best_makespan_ns));
}

/* The plan came up short. Move the fetch that caused it, or make room for
 * it and reduce again; if neither is possible the candidate is finished. */
static StageOutcome search_repair(CandidateSearch *search) {
    trace_repair(search);
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic = search->diagnostic;
    if (may_repair_again(search->options, diagnostic)) {
        ShadowSpillFetchTriggerConstraint constraint = {0};
        const int delayed = shadowspill_delay_indexed_fetch(
            &search->facts,
            &search->simulation,
            &search->workspace->schedule,
            &constraint
        );
        if (delayed < 0) {
            return search_done(search, -1);
        }
        if (delayed > 0) {
            const int recorded =
                record_fetch_constraint(search->workspace, constraint);
            if (recorded < 0) {
                return search_done(search, -1);
            }
            if (recorded > 0) {
                diagnostic->status = SHADOWSPILL_CANDIDATE_SIMULATION_INFEASIBLE;
                copy_simulation_error(diagnostic, &search->simulation);
                return search_done(search, answer_or_stop(search));
            }
            ++diagnostic->repairs.simulation_fetch_delay_attempts;
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

/* Hand the winner's schedule to the result, which owns it afterwards. */
static int adopt_selected_schedule(
    ShadowSpillPressureFitProblemResult *result,
    ShadowSpillScheduleStorage *selected
) {
    ShadowSpillIndexedSchedule *source = &selected->value;
    ShadowSpillIndexedSchedule *destination = &result->selected_schedule;
    *destination = *source;
    memset(source, 0, sizeof(*source));
    selected->action_capacity = 0U;
    selected->initial_capacity = 0U;
    selected->final_capacity = 0U;
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
    case SHADOWSPILL_STRUCT_PROBLEM_RESULT:
        return sizeof(ShadowSpillPressureFitProblemResult);
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
    free(result->resident_slice_bytes);
    free(result->alias_evict_eligible);
    memset(result, 0, sizeof(*result));
}

/*
 * Evaluating several resolved programs on one set of worker threads.
 *
 * The unit of work is a candidate of a problem -- one residency strategy, one
 * fetch rule, one coalescing mode -- and every such unit of every problem
 * competes for the same workers. Worker count and problem count are
 * independent: asking for eight workers gets eight threads whether there is
 * one problem or five, which is the whole reason this takes a list. ("Pool"
 * is deliberately not used here -- in this codebase a pool is memory, the
 * execution pool or the spill pool, and these are threads.)
 *
 * Three things are shared, and nothing else is:
 *
 * - **The task counter**, an atomic index. A worker takes the next one and
 *   owns it outright; the diagnostic it writes is that task's own slot, so
 *   diagnostics need no lock.
 * - **Each problem's winner**, behind a small spin lock. A worker takes it
 *   only when it has beaten what is recorded, which is rare, and holds it
 *   only for a schedule copy.
 * - **The placement record**, which does its own locking and is the one
 *   thing deliberately shared across problems: a plan placed under any of
 *   them bounds the search under all of them.
 *
 * Everything else a worker touches is its own workspace, including the three
 * memo tables. They are scratch for the search that worker is doing, so they
 * need no synchronisation and cannot race. A workspace is sized for one
 * problem, so a worker that moves to a different problem rebuilds it; tasks
 * are handed out problem by problem, so that is rare.
 */
/*
 * A search has two drivers and one thing they share.
 *
 *   ProgramSearch     drives the whole Program: every resolved problem, the
 *                     task counter workers pull from, and the options
 *   CandidateSearch   drives one candidate: emit -> simulate -> repair
 *   SearchedProblem   drives nothing. It is one resolved problem while it is
 *                     being searched: the facts its candidates share and the
 *                     problem itself (read-only for the whole search), its
 *                     slice of the task range, which is how one flat counter
 *                     serves several problems, and the best plan any worker
 *                     has placed for it, which is the only part that changes
 *                     and the only part that needs a lock.
 */
typedef struct SearchedProblem {
    const ShadowSpillPressureFitProblem *problem;
    /* A caller may hand in a residency problem without its sparse anchor and
     * reservation lists; the search then derives them once, here, and works
     * from its own copy of the problem. */
    ShadowSpillPressureFitProblem owned_problem;
    ShadowSpillResidencyProblem owned_residency;
    ShadowSpillResidencySparseLists lists;
    int derived;
    ShadowSpillScheduleFacts facts;
    ShadowSpillPressureFitProblemResult *result;
    /* Global index of this problem's first candidate. */
    uint32_t first_task;
    uint32_t candidate_count;
    /* The best plan any worker placed for this problem, and its schedule.
     * Guarded because several workers can beat it at once. */
    ShadowSpillScheduleStorage selected;
    uint64_t selected_makespan_ns;
    uint32_t selected_candidate;
    atomic_flag guard;
    int ready;
} SearchedProblem;

typedef struct ProgramSearch {
    const ShadowSpillPressureFitProblemOptions *options;
    SearchedProblem *problems;
    uint32_t problem_count;
    uint32_t total_tasks;
    _Atomic uint32_t next_task;
    /* The clock every reported timestamp is measured from, so a caller
     * reading several problems out of one call sees one timeline. */
    uint64_t origin_ns;
} ProgramSearch;

typedef struct SearchWorker {
    ProgramSearch *search;
    CandidateWorkspace workspace;
    /* Which problem the workspace is sized for, or NO_INDEX before the first
     * task. Sizes come from the problem, so this cannot be shared across
     * problems without rebuilding. */
    uint32_t workspace_problem;
    int failed;
} SearchWorker;

/* Which resolved problem owns a global task index, and which of its
 * candidates the task names. */
static uint32_t problem_of_task(
    const ProgramSearch *search, uint32_t task, uint32_t *candidate
) {
    for (uint32_t index = 0U; index < search->problem_count; ++index) {
        const SearchedProblem *problem = &search->problems[index];
        if (task < problem->first_task + problem->candidate_count) {
            *candidate = task - problem->first_task;
            return index;
        }
    }
    *candidate = 0U;
    return SHADOWSPILL_PLANNER_NO_INDEX;
}

/* Give this worker a workspace sized for `index`, reusing the one it has when
 * it is already for that problem. */
static int worker_workspace_for(SearchWorker *worker, uint32_t index) {
    if (worker->workspace_problem == index) {
        return 0;
    }
    if (worker->workspace_problem != SHADOWSPILL_PLANNER_NO_INDEX) {
        candidate_workspace_destroy(&worker->workspace);
        worker->workspace_problem = SHADOWSPILL_PLANNER_NO_INDEX;
    }
    if (candidate_workspace_create(
            worker->search->problems[index].problem, &worker->workspace
        ) != 0) {
        return -1;
    }
    worker->workspace_problem = index;
    return 0;
}

/* Record a plan as this problem's answer when it beats what is held. */
static int offer_problem_winner(
    SearchedProblem *problem,
    uint32_t candidate,
    uint64_t makespan_ns,
    const ShadowSpillScheduleStorage *schedule
) {
    while (atomic_flag_test_and_set_explicit(&problem->guard, memory_order_acquire)) {
        /* Held only for a schedule copy, and taken only by a worker that has
         * already beaten the record, so spinning beats descheduling. */
        shadowspill_thread_yield();
    }
    int failed = 0;
    if (problem->selected_candidate == SHADOWSPILL_PLANNER_NO_INDEX ||
        makespan_ns < problem->selected_makespan_ns ||
        (makespan_ns == problem->selected_makespan_ns &&
         candidate < problem->selected_candidate)) {
        failed = shadowspill_schedule_storage_copy(
            &problem->selected, schedule
        ) != 0;
        if (!failed) {
            problem->selected_candidate = candidate;
            problem->selected_makespan_ns = makespan_ns;
        }
    }
    atomic_flag_clear_explicit(&problem->guard, memory_order_release);
    return failed ? -1 : 0;
}

/* Stamp what a task cost and when it ended. Every exit that answered goes
 * through here, so a candidate that stopped early still lands on the
 * timeline and still reports the work it did before it stopped. An exit
 * that failed internally does not: it ends the whole search, and there is
 * no candidate left to describe. */
static void finish_task(
    ShadowSpillPressureFitCandidateDiagnostic *diagnostic,
    const ProgramSearch *search,
    CandidateWorkspace *workspace,
    ShadowSpillPressureFitWorkDiagnostics before,
    uint64_t started
) {
    diagnostic->work = work_delta(workspace_work(workspace), before);
    section_close_total(&diagnostic->work.sections, started);
    /* Both ends are stamped here, at the end, because evaluating a candidate
     * initializes the diagnostic again and would erase a start written
     * before it. */
    diagnostic->started_ns = started - search->origin_ns;
    diagnostic->finished_ns = shadowspill_monotonic_ns() - search->origin_ns;
}

/* Evaluate one candidate. Returns -1 only for an internal failure; a
 * candidate that simply has no answer returns 0. */
static int worker_evaluate_task(SearchWorker *worker, uint32_t task) {
    ProgramSearch *search = worker->search;
    uint32_t candidate = 0U;
    const uint32_t index = problem_of_task(search, task, &candidate);
    if (index == SHADOWSPILL_PLANNER_NO_INDEX) {
        return -1;
    }
    if (worker_workspace_for(worker, index) != 0) {
        return -1;
    }
    SearchedProblem *problem = &search->problems[index];
    CandidateWorkspace *workspace = &worker->workspace;
    const ShadowSpillPressureFitProblemOptions *options = search->options;

    const uint32_t modes = options->coalescing_mode_count;
    const uint32_t rules = options->fetch_rule_count;
    const uint8_t mode = options->coalescing_modes[candidate % modes];
    const uint8_t rule = options->fetch_rules[(candidate / modes) % rules];
    const uint8_t strategy =
        options->residency_strategies[candidate / (modes * rules)];

    ShadowSpillPressureFitCandidateDiagnostic *diagnostic =
        &problem->result->candidates[candidate];
    initialize_diagnostic(diagnostic, strategy, rule, mode);

    /* Everything from here on is this candidate's, the base reduction
     * included: it is work a worker does for this task and nobody else's. */
    const ShadowSpillPressureFitWorkDiagnostics before = workspace_work(workspace);
    const uint64_t started = shadowspill_monotonic_ns();

    /* This strategy's base residency, from this worker's own memo. */
    const uint64_t pressure_cells =
        (uint64_t)problem->problem->residency->device_count *
        problem->problem->residency->boundary_count;
    memset(
        workspace->extra_pressure,
        0,
        (size_t)pressure_cells * sizeof(*workspace->extra_pressure)
    );
    ShadowSpillResidencyOptions reduce_options;
    residency_options(workspace, strategy, &reduce_options);
    ShadowSpillResidencyResult base;
    const Section reduce = section_open(&workspace->sections.reduce_ns);
    const ShadowSpillStatus base_status = reduce_residency(
        problem->problem,
        workspace,
        &reduce_options,
        strategy,
        workspace->base_resident,
        workspace->base_breaks,
        &base
    );
    section_close(reduce);
    workspace->base_residency = workspace->current_residency;
    if (base_status == SHADOWSPILL_STATUS_ANALYTIC_INFEASIBLE) {
        copy_analytic_error(diagnostic, &base);
        finish_task(diagnostic, search, workspace, before, started);
        return 0;
    }
    if (base_status != SHADOWSPILL_STATUS_OK) {
        return -1;
    }

    const int valid = evaluate_candidate(
        problem->problem,
        &problem->facts,
        options,
        workspace,
        strategy,
        rule,
        mode,
        diagnostic
    );
    finish_task(diagnostic, search, workspace, before, started);
    if (valid < 0) {
        return -1;
    }
    if (valid > 0) {
        return offer_problem_winner(
            problem, candidate, diagnostic->makespan_ns, &workspace->schedule
        );
    }
    return 0;
}

static void *worker_main(void *argument) {
    SearchWorker *worker = argument;
    shadowspill_name_current_thread("shadowspill.plan");
    while (worker->failed == 0) {
        const uint32_t task = atomic_fetch_add_explicit(
            &worker->search->next_task, 1U, memory_order_relaxed
        );
        if (task >= worker->search->total_tasks) {
            break;
        }
        if (worker_evaluate_task(worker, task) != 0) {
            worker->failed = 1;
        }
    }
    return NULL;
}

/* How many threads to evaluate with. Scheduling rather than search, so this
 * is free to consider the machine: it changes neither which plans are legal
 * nor how they simulate. It does change how many candidates the shared
 * record lets a search skip, which is why per-candidate counters move with
 * it. Never more threads than there is work to give them. */
static uint32_t worker_count_for(
    const ShadowSpillPressureFitProblemOptions *options, uint32_t tasks
) {
    if (tasks <= 1U || options->workers == 1U) {
        return 1U;
    }
    const uint32_t wanted = options->workers != 0U
        ? options->workers
        : shadowspill_logical_cpu_count();
    return wanted < tasks ? wanted : tasks;
}

/* Release everything the search allocated, whatever it managed to finish. */
static void program_search_destroy(
    ProgramSearch *search, SearchWorker *workers, uint32_t worker_count
) {
    for (uint32_t index = 0U; index < worker_count; ++index) {
        if (workers[index].workspace_problem != SHADOWSPILL_PLANNER_NO_INDEX) {
            candidate_workspace_destroy(&workers[index].workspace);
        }
    }
    free(workers);
    for (uint32_t index = 0U; index < search->problem_count; ++index) {
        SearchedProblem *problem = &search->problems[index];
        if (problem->ready) {
            shadowspill_schedule_facts_destroy(&problem->facts);
            shadowspill_schedule_storage_destroy(&problem->selected);
        }
        if (problem->derived) {
            shadowspill_residency_sparse_lists_destroy(&problem->lists);
        }
    }
    free(search->problems);
}

ShadowSpillStatus shadowspill_evaluate_pressurefit_problems(
    const ShadowSpillPressureFitProblem *problems,
    uint32_t problem_count,
    const ShadowSpillPressureFitProblemOptions *options,
    ShadowSpillPressureFitProblemResult *results
) {
    if (problems == NULL || results == NULL || problem_count == 0U) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }
    for (uint32_t index = 0U; index < problem_count; ++index) {
        memset(&results[index], 0, sizeof(results[index]));
        results[index].selected_candidate_index = SHADOWSPILL_PLANNER_NO_INDEX;
        results[index].status = SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        if (!problem_valid(&problems[index], options)) {
            return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
        }
    }
    uint32_t per_problem = 0U;
    if (multiply_u32(
            options->residency_strategy_count,
            options->fetch_rule_count,
            &per_problem
        ) != 0 ||
        multiply_u32(per_problem, options->coalescing_mode_count, &per_problem) != 0) {
        return SHADOWSPILL_STATUS_INVALID_ARGUMENT;
    }

    const uint64_t started = shadowspill_monotonic_ns();
    ProgramSearch search = {
        .options = options,
        .problem_count = problem_count,
        .origin_ns = started,
    };
    search.problems = calloc(problem_count, sizeof(*search.problems));
    if (search.problems == NULL) {
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    atomic_init(&search.next_task, 0U);

    for (uint32_t index = 0U; index < problem_count; ++index) {
        SearchedProblem *problem = &search.problems[index];
        problem->problem = &problems[index];
        if (problems[index].residency->anchor_offsets == NULL) {
            const ShadowSpillResidencyProblem *residency = problems[index].residency;
            if (shadowspill_residency_sparse_lists_build(
                    residency->anchors,
                    residency->latest_access_task,
                    residency->output_reservations,
                    residency->alias_count,
                    residency->boundary_count,
                    &problem->lists
                ) != 0) {
                program_search_destroy(&search, NULL, 0U);
                return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
            }
            problem->derived = 1;
            problem->owned_residency = *residency;
            problem->owned_residency.anchor_offsets = problem->lists.anchor_offsets;
            problem->owned_residency.anchor_positions = problem->lists.anchor_positions;
            problem->owned_residency.anchor_tasks = problem->lists.anchor_tasks;
            problem->owned_residency.reserved_offsets = problem->lists.reserved_offsets;
            problem->owned_residency.reserved_positions = problem->lists.reserved_positions;
            problem->owned_problem = problems[index];
            problem->owned_problem.residency = &problem->owned_residency;
            problem->problem = &problem->owned_problem;
        }
        problem->result = &results[index];
        problem->first_task = search.total_tasks;
        problem->candidate_count = per_problem;
        problem->selected_candidate = SHADOWSPILL_PLANNER_NO_INDEX;
        search.total_tasks += per_problem;
        results[index].candidates =
            calloc(per_problem, sizeof(*results[index].candidates));
        if (results[index].candidates == NULL ||
            shadowspill_schedule_facts_create(problem->problem, &problem->facts) != 0 ||
            shadowspill_schedule_storage_create(
                problem->problem->residency->alias_count, &problem->selected
            ) != 0) {
            program_search_destroy(&search, NULL, 0U);
            return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        }
        results[index].candidate_count = per_problem;
        problem->ready = 1;
    }

    const uint32_t worker_count = worker_count_for(options, search.total_tasks);
    SearchWorker *workers = calloc(worker_count, sizeof(*workers));
    if (workers == NULL) {
        program_search_destroy(&search, NULL, 0U);
        return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
    }
    for (uint32_t index = 0U; index < worker_count; ++index) {
        workers[index].search = &search;
        workers[index].workspace_problem = SHADOWSPILL_PLANNER_NO_INDEX;
    }

    /* The calling thread is one of the workers, so a single-worker run needs
     * no thread at all and the common case starts one fewer. */
    pthread_t *threads = NULL;
    uint32_t started_threads = 0U;
    if (worker_count > 1U) {
        threads = calloc(worker_count - 1U, sizeof(*threads));
        if (threads == NULL) {
            program_search_destroy(&search, workers, worker_count);
            return SHADOWSPILL_STATUS_INTERNAL_FAILURE;
        }
        for (uint32_t index = 1U; index < worker_count; ++index) {
            if (pthread_create(
                    &threads[index - 1U], NULL, worker_main, &workers[index]
                ) != 0) {
                break;
            }
            ++started_threads;
        }
    }
    worker_main(&workers[0]);
    for (uint32_t index = 0U; index < started_threads; ++index) {
        (void)pthread_join(threads[index], NULL);
    }
    free(threads);

    int failed = 0;
    for (uint32_t index = 0U; index < worker_count; ++index) {
        failed |= workers[index].failed;
    }

    /* Adopt each problem's winner, and sum what its candidates did. */
    for (uint32_t index = 0U; index < problem_count; ++index) {
        SearchedProblem *problem = &search.problems[index];
        ShadowSpillPressureFitProblemResult *result = &results[index];
        for (uint32_t slot = 0U; slot < problem->candidate_count; ++slot) {
            const ShadowSpillPressureFitCandidateDiagnostic *candidate =
                &result->candidates[slot];
            add_repairs(&result->repairs, &candidate->repairs);
            result->work = add_work(result->work, candidate->work);
            /* The problem spans its candidates. A candidate no worker
             * reached has both stamps zero and is skipped, so an untouched
             * problem keeps the zero span it started with. */
            if (candidate->finished_ns == 0U) {
                continue;
            }
            if (result->finished_ns == 0U ||
                candidate->started_ns < result->started_ns) {
                result->started_ns = candidate->started_ns;
            }
            if (candidate->finished_ns > result->finished_ns) {
                result->finished_ns = candidate->finished_ns;
            }
        }
        if (failed) {
            result->status = SHADOWSPILL_STATUS_PLANNER_INTERNAL_ERROR;
            continue;
        }
        if (problem->selected_candidate == SHADOWSPILL_PLANNER_NO_INDEX) {
            result->status = SHADOWSPILL_STATUS_NO_FEASIBLE_CANDIDATE;
            continue;
        }
        result->selected_candidate_index = problem->selected_candidate;
        result->selected_makespan_ns = problem->selected_makespan_ns;
        if (adopt_selected_schedule(result, &problem->selected) != 0) {
            result->status = SHADOWSPILL_STATUS_INTERNAL_FAILURE;
            failed = 1;
            continue;
        }
        result->status = SHADOWSPILL_STATUS_OK;
    }

    /* A problem's sections are the sum of its candidates', including their
     * totals and residuals, so the identity total == named + residual still
     * holds -- as an accounting identity over work done, which is what it has
     * to be once several workers run at once. Wall time is no longer that
     * sum and is not reported here: the whole point of the workers is that
     * the call finishes sooner than the work it did. */
    (void)started;
    program_search_destroy(&search, workers, worker_count);
    if (failed) {
        return SHADOWSPILL_STATUS_PLANNER_INTERNAL_ERROR;
    }
    for (uint32_t index = 0U; index < problem_count; ++index) {
        if (results[index].status == SHADOWSPILL_STATUS_OK) {
            return SHADOWSPILL_STATUS_OK;
        }
    }
    return SHADOWSPILL_STATUS_NO_FEASIBLE_CANDIDATE;
}
