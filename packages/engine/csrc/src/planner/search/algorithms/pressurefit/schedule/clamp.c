/* Moving triggers until the span they fill fits the budget. */
#include "internal.h"

/*
 * Packed-fit clamping.
 *
 * Packing fetches as early as their windows allow can put more bytes at a
 * boundary than it holds. Clamping walks the boundaries in order and delays
 * fetches -- cheapest to give up first -- until each one fits or nothing is
 * left to delay. What it needs while it runs lives in `Clamp`.
 *
 * The fetches a boundary can still give up are kept as a sweep: a fetch
 * joins when the walk reaches its trigger, waits in a heap ordered by the
 * rank it is given up in, and a fetch that is delayed joins again at its new
 * trigger. An entry whose fetch has since moved, reached its latest trigger,
 * or whose window the walk has left is stale, and is dropped when it
 * surfaces. Nothing here is sized by alias and boundary together.
 */
typedef struct HeapEntry {
    uint32_t rank;
    uint32_t reload;
    /* The trigger the fetch had when it joined; a different one now means
     * it was delayed and joined again elsewhere. */
    uint32_t trigger;
} HeapEntry;

typedef struct Clamp {
    /* Bytes charged to each device at each boundary. */
    uint64_t *used;
    /* The order fetches are given up in. */
    ReloadRank *ranked;
    uint32_t *rank_by_reload;
    /* The fetches joining the sweep at each boundary, as a list per boundary
     * threaded through `node_next`. */
    uint32_t *bucket_head;
    uint32_t *node_reload;
    uint32_t *node_next;
    uint32_t node_count;
    uint32_t node_capacity;
    HeapEntry *heap;
    uint32_t heap_count;
    uint32_t heap_capacity;
    /* Entries set aside while choosing at one boundary: fetches whose alias
     * is resident there anyway, which the boundary cannot give up but a
     * later one can. */
    HeapEntry *aside;
    uint32_t aside_count;
    uint32_t aside_capacity;
} Clamp;

static void clamp_destroy(Clamp *clamp) {
    free(clamp->used);
    free(clamp->ranked);
    free(clamp->rank_by_reload);
    free(clamp->bucket_head);
    free(clamp->node_reload);
    free(clamp->node_next);
    free(clamp->heap);
    free(clamp->aside);
    memset(clamp, 0, sizeof(*clamp));
}

static int grow(void **values, uint32_t *capacity, uint32_t needed, size_t element) {
    if (needed <= *capacity) {
        return 0;
    }
    uint32_t selected = *capacity == 0U ? 64U : *capacity;
    while (selected < needed) {
        if (selected > UINT32_MAX / 2U) {
            selected = needed;
            break;
        }
        selected *= 2U;
    }
    void *replacement = realloc(*values, (size_t)selected * element);
    if (replacement == NULL) {
        return -1;
    }
    *values = replacement;
    *capacity = selected;
    return 0;
}

static int clamp_create(
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillResidency *residency,
    uint32_t reload_count,
    int fetch_headroom,
    Clamp *clamp
) {
    memset(clamp, 0, sizeof(*clamp));
    size_t pressure_cells = 0U;
    if (shadowspill_schedule_checked_cells(
            facts->device_count, facts->boundary_count, &pressure_cells
        ) != 0) {
        return -1;
    }
    const size_t reload_slots = reload_count == 0U ? 1U : (size_t)reload_count;
    const size_t task_slots = facts->task_count == 0U ? 1U : facts->task_count;
    clamp->used =
        calloc(pressure_cells == 0U ? 1U : pressure_cells, sizeof(*clamp->used));
    clamp->ranked = malloc(reload_slots * sizeof(*clamp->ranked));
    clamp->rank_by_reload = malloc(reload_slots * sizeof(*clamp->rank_by_reload));
    clamp->bucket_head = malloc(task_slots * sizeof(*clamp->bucket_head));
    if (clamp->used == NULL || clamp->ranked == NULL ||
        clamp->rank_by_reload == NULL || clamp->bucket_head == NULL ||
        grow((void **)&clamp->node_reload, &clamp->node_capacity, reload_count,
             sizeof(*clamp->node_reload)) != 0 ||
        shadowspill_schedule_build_pressure(facts, residency, fetch_headroom, clamp->used) != 0) {
        return -1;
    }
    /* The two node arrays grow together; `node_next` follows the capacity
     * `node_reload` reached. */
    clamp->node_next = malloc((size_t)clamp->node_capacity * sizeof(*clamp->node_next));
    return clamp->node_next == NULL ? -1 : 0;
}

/* One fetch's window, for checking that an alias's windows are disjoint. */
typedef struct Window {
    uint32_t alias;
    uint32_t trigger;
    uint32_t entry_boundary;
} Window;

static int window_compare(const void *left_value, const void *right_value) {
    const Window *left = left_value;
    const Window *right = right_value;
    if (left->alias != right->alias) {
        return left->alias < right->alias ? -1 : 1;
    }
    if (left->trigger != right->trigger) {
        return left->trigger < right->trigger ? -1 : 1;
    }
    return 0;
}

/* Charge every boundary each fetch's window covers, skipping boundaries the
 * object is resident at anyway.
 *
 * One alias's windows never overlap: a span's fetch may start no earlier
 * than the previous span's departure, which is at or after that span's
 * entry boundary, where the previous window ends. So a boundary is charged
 * by at most one window per alias and is charged once by charging each
 * window. That is checked rather than assumed, because the charge would be
 * silently wrong without it. */
static int charge_fetch_windows(
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillResidency *residency,
    const Reload *reloads,
    uint32_t reload_count,
    Clamp *clamp
) {
    const ShadowSpillPressureFitResidencyProblem *problem = facts->problem->residency;
    Window *windows = malloc((reload_count == 0U ? 1U : (size_t)reload_count) * sizeof(*windows));
    if (windows == NULL) {
        return -1;
    }
    for (uint32_t index = 0U; index < reload_count; ++index) {
        windows[index] = (Window){
            .alias = reloads[index].alias,
            .trigger = reloads[index].trigger,
            .entry_boundary = reloads[index].entry_boundary,
        };
    }
    qsort(windows, reload_count, sizeof(*windows), window_compare);
    for (uint32_t position = 1U; position < reload_count; ++position) {
        const Window *previous = &windows[position - 1U];
        const Window *current = &windows[position];
        if (previous->alias == current->alias &&
            previous->trigger < previous->entry_boundary &&
            current->trigger < previous->entry_boundary) {
            free(windows);
            return -1;
        }
    }
    free(windows);
    for (uint32_t index = 0U; index < reload_count; ++index) {
        const Reload *reload = &reloads[index];
        clamp->ranked[index] = (ReloadRank){
            .index = index,
            .entry_boundary = reload->entry_boundary,
            .size_bytes = problem->alias_size_bytes[reload->alias],
            .alias = reload->alias,
        };
        const uint32_t device = problem->alias_device[reload->alias];
        for (uint32_t boundary = reload->trigger;
             boundary < reload->entry_boundary;
             ++boundary) {
            if (shadowspill_residency_resident_at(residency, reload->alias, boundary + 1U)) {
                continue;
            }
            clamp->used[(uint64_t)device * facts->boundary_count + boundary + 1U] +=
                problem->alias_size_bytes[reload->alias];
        }
    }
    return 0;
}

static void rank_reloads(uint32_t reload_count, Clamp *clamp) {
    qsort(clamp->ranked, reload_count, sizeof(*clamp->ranked), shadowspill_schedule_reload_rank_compare);
    for (uint32_t rank = 0U; rank < reload_count; ++rank) {
        clamp->rank_by_reload[clamp->ranked[rank].index] = rank;
    }
}

/* A fetch joins the sweep at `boundary`. */
static int enter_at(Clamp *clamp, uint32_t boundary, uint32_t reload) {
    if (clamp->node_count == clamp->node_capacity) {
        const uint32_t before = clamp->node_capacity;
        if (grow((void **)&clamp->node_reload, &clamp->node_capacity,
                 clamp->node_count + 1U, sizeof(*clamp->node_reload)) != 0) {
            return -1;
        }
        uint32_t *next = realloc(
            clamp->node_next, (size_t)clamp->node_capacity * sizeof(*clamp->node_next)
        );
        if (next == NULL) {
            clamp->node_capacity = before;
            return -1;
        }
        clamp->node_next = next;
    }
    const uint32_t node = clamp->node_count++;
    clamp->node_reload[node] = reload;
    clamp->node_next[node] = clamp->bucket_head[boundary];
    clamp->bucket_head[boundary] = node;
    return 0;
}

static int heap_push(HeapEntry **heap, uint32_t *count, uint32_t *capacity, HeapEntry entry) {
    if (grow((void **)heap, capacity, *count + 1U, sizeof(**heap)) != 0) {
        return -1;
    }
    HeapEntry *entries = *heap;
    uint32_t child = (*count)++;
    while (child != 0U) {
        const uint32_t parent = (child - 1U) / 2U;
        if (entries[parent].rank <= entry.rank) {
            break;
        }
        entries[child] = entries[parent];
        child = parent;
    }
    entries[child] = entry;
    return 0;
}

static void heap_pop(HeapEntry *entries, uint32_t *count) {
    const uint32_t remaining = --*count;
    if (remaining == 0U) {
        return;
    }
    const HeapEntry moved = entries[remaining];
    uint32_t parent = 0U;
    while (1) {
        const uint32_t left = parent * 2U + 1U;
        if (left >= remaining) {
            break;
        }
        const uint32_t right = left + 1U;
        uint32_t best = left;
        if (right < remaining && entries[right].rank < entries[left].rank) {
            best = right;
        }
        if (entries[best].rank >= moved.rank) {
            break;
        }
        entries[parent] = entries[best];
        parent = best;
    }
    entries[parent] = moved;
}

/* Every fetch that can still move joins at its trigger. Called once per
 * device pass, from the triggers as they stand. */
static int seed_sweep(const Reload *reloads, uint32_t reload_count, uint32_t task_count, Clamp *clamp) {
    for (uint32_t boundary = 0U; boundary < task_count; ++boundary) {
        clamp->bucket_head[boundary] = UINT32_MAX;
    }
    clamp->node_count = 0U;
    clamp->heap_count = 0U;
    for (uint32_t index = 0U; index < reload_count; ++index) {
        const Reload *reload = &reloads[index];
        if (reload->trigger < reload->latest_trigger && reload->trigger < task_count &&
            enter_at(clamp, reload->trigger, index) != 0) {
            return -1;
        }
    }
    return 0;
}

/* The fetches joining at `boundary` take their place in the heap. */
static int admit_entrants(const Reload *reloads, uint32_t boundary, Clamp *clamp) {
    for (uint32_t node = clamp->bucket_head[boundary]; node != UINT32_MAX;
         node = clamp->node_next[node]) {
        const uint32_t reload = clamp->node_reload[node];
        if (reloads[reload].trigger != boundary) {
            continue;
        }
        if (heap_push(
                &clamp->heap, &clamp->heap_count, &clamp->heap_capacity,
                (HeapEntry){clamp->rank_by_reload[reload], reload, boundary}
            ) != 0) {
            return -1;
        }
    }
    return 0;
}

/* The fetch `boundary` gives up first: the lowest rank among those whose
 * window covers the boundary, that can still move, and whose alias is not
 * resident there anyway. Stale entries are dropped on the way; entries
 * skipped for residency go back afterwards. Returns the reload index, or
 * UINT32_MAX when the boundary has nothing to give up; -1 is reported
 * through `failed`. */
static uint32_t first_movable(
    const ShadowSpillResidency *residency,
    const Reload *reloads,
    uint32_t boundary,
    Clamp *clamp,
    int *failed
) {
    uint32_t selected = UINT32_MAX;
    clamp->aside_count = 0U;
    while (clamp->heap_count != 0U) {
        const HeapEntry top = clamp->heap[0];
        const Reload *reload = &reloads[top.reload];
        if (reload->trigger != top.trigger ||
            reload->trigger >= reload->latest_trigger ||
            boundary >= reload->entry_boundary) {
            heap_pop(clamp->heap, &clamp->heap_count);
            continue;
        }
        if (shadowspill_residency_resident_at(residency, reload->alias, boundary + 1U)) {
            heap_pop(clamp->heap, &clamp->heap_count);
            if (grow((void **)&clamp->aside, &clamp->aside_capacity,
                     clamp->aside_count + 1U, sizeof(*clamp->aside)) != 0) {
                *failed = 1;
                return UINT32_MAX;
            }
            clamp->aside[clamp->aside_count++] = top;
            continue;
        }
        selected = top.reload;
        break;
    }
    for (uint32_t index = 0U; index < clamp->aside_count; ++index) {
        if (heap_push(&clamp->heap, &clamp->heap_count, &clamp->heap_capacity,
                      clamp->aside[index]) != 0) {
            *failed = 1;
            return UINT32_MAX;
        }
    }
    return selected;
}

/* Move one fetch later, and give back the pressure the window it vacated
 * was holding. Its heap entry is left behind as stale; it joins again at
 * the new trigger when that is not its last. */
static int delay_reload(
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillResidency *residency,
    Reload *reload,
    uint32_t index,
    uint32_t boundary,
    Clamp *clamp
) {
    const ShadowSpillPressureFitResidencyProblem *problem = facts->problem->residency;
    const uint32_t old_trigger = reload->trigger;
    uint32_t new_trigger = boundary + 1U;
    if (new_trigger > reload->latest_trigger) {
        new_trigger = reload->latest_trigger;
    }
    reload->trigger = new_trigger;
    for (uint32_t retired = old_trigger; retired < new_trigger; ++retired) {
        if (shadowspill_residency_resident_at(residency, reload->alias, retired + 1U)) {
            continue;
        }
        const uint32_t device = problem->alias_device[reload->alias];
        clamp->used[(uint64_t)device * facts->boundary_count + retired + 1U] -=
            problem->alias_size_bytes[reload->alias];
    }
    if (new_trigger < reload->latest_trigger && new_trigger < facts->task_count) {
        return enter_at(clamp, new_trigger, index);
    }
    return 0;
}

/* Delay fetches at one boundary until it fits. Running out of fetches to
 * delay is not an error: the boundary is as relieved as moving fetches can
 * make it, and what remains is the reducer's problem rather than this one. */
static int relieve_boundary(
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillResidency *residency,
    Reload *reloads,
    uint32_t device,
    uint32_t boundary,
    Clamp *clamp
) {
    const ShadowSpillPressureFitResidencyProblem *problem = facts->problem->residency;
    const uint64_t position =
        (uint64_t)device * facts->boundary_count + boundary + 1U;
    while (clamp->used[position] >
           shadowspill_boundary_capacity(problem, device, boundary + 1U)) {
        int failed = 0;
        const uint32_t selected = first_movable(
            residency, reloads, boundary, clamp, &failed
        );
        if (failed) {
            return -1;
        }
        if (selected == UINT32_MAX) {
            return 0;
        }
        if (delay_reload(facts, residency, &reloads[selected], selected, boundary, clamp) != 0) {
            return -1;
        }
    }
    return 0;
}

int shadowspill_schedule_clamp_triggers_to_fit(
    const ShadowSpillScheduleFacts *facts,
    const ShadowSpillResidency *residency,
    Reload *reloads,
    uint32_t reload_count,
    int fetch_headroom
) {
    Clamp clamp;
    if (clamp_create(
            facts, residency, reload_count, fetch_headroom, &clamp
        ) != 0 ||
        charge_fetch_windows(facts, residency, reloads, reload_count, &clamp) != 0) {
        clamp_destroy(&clamp);
        return -1;
    }
    rank_reloads(reload_count, &clamp);
    for (uint32_t device = 0U; device < facts->device_count; ++device) {
        if (seed_sweep(reloads, reload_count, facts->task_count, &clamp) != 0) {
            clamp_destroy(&clamp);
            return -1;
        }
        for (uint32_t boundary = 0U; boundary < facts->task_count; ++boundary) {
            if (admit_entrants(reloads, boundary, &clamp) != 0 ||
                relieve_boundary(facts, residency, reloads, device, boundary, &clamp) != 0) {
                clamp_destroy(&clamp);
                return -1;
            }
        }
    }
    clamp_destroy(&clamp);
    return 0;
}
