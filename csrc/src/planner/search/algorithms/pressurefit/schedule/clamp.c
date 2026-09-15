/* Moving triggers until the span they fill fits the budget. */
#include "internal.h"

/*
 * Packed-fit clamping.
 *
 * Packing fetches as early as their windows allow can put more bytes at a
 * boundary than it holds. Clamping walks the boundaries that overflow and
 * delays fetches -- cheapest to give up first -- until each one fits or
 * nothing is left to delay. What it needs while it runs lives in `Clamp`.
 */
typedef struct Clamp {
    /* Bytes charged to each device at each boundary. */
    uint64_t *used;
    /* How many fetch windows cover each alias/boundary pair, so a boundary is
     * charged once however many fetches cover it. */
    uint32_t *counts;
    /* Bitset per boundary: which fetches still have room to move later. */
    uint64_t *active;
    /* The order fetches are given up in. */
    ReloadRank *ranked;
    uint32_t *rank_by_reload;
    uint32_t word_count;
} Clamp;

static void clamp_destroy(Clamp *clamp) {
    free(clamp->used);
    free(clamp->counts);
    free(clamp->active);
    free(clamp->ranked);
    free(clamp->rank_by_reload);
    memset(clamp, 0, sizeof(*clamp));
}

static int clamp_create(
    const ShadowSpillScheduleFacts *facts,
    const uint8_t *resident,
    const uint8_t *breaks,
    uint32_t reload_count,
    int fetch_headroom,
    Clamp *clamp
) {
    memset(clamp, 0, sizeof(*clamp));
    clamp->word_count = reload_count / 64U + (reload_count % 64U != 0U);
    size_t pressure_cells = 0U;
    size_t active_words = 0U;
    if (shadowspill_schedule_checked_cells(
            facts->device_count, facts->boundary_count, &pressure_cells
        ) != 0 ||
        shadowspill_schedule_checked_cells(clamp->word_count, facts->task_count, &active_words) != 0) {
        return -1;
    }
    const size_t reload_slots = reload_count == 0U ? 1U : (size_t)reload_count;
    clamp->used =
        calloc(pressure_cells == 0U ? 1U : pressure_cells, sizeof(*clamp->used));
    clamp->counts = calloc(
        (size_t)facts->alias_count *
            (facts->task_count == 0U ? 1U : facts->task_count),
        sizeof(*clamp->counts)
    );
    clamp->active =
        calloc(active_words == 0U ? 1U : active_words, sizeof(*clamp->active));
    clamp->ranked = malloc(reload_slots * sizeof(*clamp->ranked));
    clamp->rank_by_reload = malloc(reload_slots * sizeof(*clamp->rank_by_reload));
    if (clamp->used == NULL || clamp->counts == NULL || clamp->active == NULL ||
        clamp->ranked == NULL || clamp->rank_by_reload == NULL ||
        shadowspill_schedule_build_pressure(facts, resident, breaks, fetch_headroom, clamp->used) != 0) {
        return -1;
    }
    return 0;
}

/* Charge every boundary each fetch's window covers, skipping boundaries the
 * object is resident at anyway. */
static void charge_fetch_windows(
    const ShadowSpillScheduleFacts *facts,
    const uint8_t *resident,
    const Reload *reloads,
    uint32_t reload_count,
    Clamp *clamp
) {
    const ShadowSpillPressureFitResidencyProblem *problem = facts->problem->residency;
    for (uint32_t index = 0U; index < reload_count; ++index) {
        const Reload *reload = &reloads[index];
        clamp->ranked[index] = (ReloadRank){
            .index = index,
            .entry_boundary = reload->entry_boundary,
            .size_bytes = problem->alias_size_bytes[reload->alias],
            .alias = reload->alias,
        };
        for (uint32_t boundary = reload->trigger;
             boundary < reload->entry_boundary;
             ++boundary) {
            if (shadowspill_cell_get(
                    resident,
                    shadowspill_schedule_cell(reload->alias, facts->boundary_count, boundary + 1U)
                )) {
                continue;
            }
            const uint64_t position =
                (uint64_t)reload->alias * facts->task_count + boundary;
            if (clamp->counts[position]++ == 0U) {
                const uint32_t device = problem->alias_device[reload->alias];
                clamp->used[(uint64_t)device * facts->boundary_count +
                            boundary + 1U] +=
                    problem->alias_size_bytes[reload->alias];
            }
        }
    }
}

static void rank_reloads(uint32_t reload_count, Clamp *clamp) {
    qsort(clamp->ranked, reload_count, sizeof(*clamp->ranked), shadowspill_schedule_reload_rank_compare);
    for (uint32_t rank = 0U; rank < reload_count; ++rank) {
        clamp->rank_by_reload[clamp->ranked[rank].index] = rank;
    }
}

/* Mark the fetches a boundary could still give up. One already at its latest
 * trigger has nowhere to go and is never marked. */
static void mark_movable_reloads(
    const ShadowSpillScheduleFacts *facts,
    const uint8_t *resident,
    const Reload *reloads,
    uint32_t reload_count,
    Clamp *clamp
) {
    for (uint32_t index = 0U; index < reload_count; ++index) {
        const Reload *reload = &reloads[index];
        if (reload->trigger >= reload->latest_trigger) {
            continue;
        }
        const uint32_t rank = clamp->rank_by_reload[index];
        const uint64_t mask = UINT64_C(1) << (rank & 63U);
        const uint32_t word = rank >> 6U;
        for (uint32_t boundary = reload->trigger;
             boundary < reload->entry_boundary;
             ++boundary) {
            if (!shadowspill_cell_get(
                    resident,
                    shadowspill_schedule_cell(reload->alias, facts->boundary_count, boundary + 1U)
                )) {
                clamp->active[(uint64_t)boundary * clamp->word_count + word] |= mask;
            }
        }
    }
}

/* Move one fetch later, and give back the pressure the window it vacated
 * was holding. */
static void delay_reload(
    const ShadowSpillScheduleFacts *facts,
    const uint8_t *resident,
    Reload *reload,
    uint32_t rank,
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
    shadowspill_schedule_clear_active_reload(
        clamp->active, clamp->word_count, rank, old_trigger, new_trigger
    );
    if (new_trigger == reload->latest_trigger) {
        /* Out of room to move, so it can never be chosen again. */
        shadowspill_schedule_clear_active_reload(
            clamp->active, clamp->word_count, rank, new_trigger, facts->task_count
        );
    }
    for (uint32_t retired = old_trigger; retired < new_trigger; ++retired) {
        if (shadowspill_cell_get(
                resident,
                shadowspill_schedule_cell(reload->alias, facts->boundary_count, retired + 1U)
            )) {
            continue;
        }
        const uint64_t position =
            (uint64_t)reload->alias * facts->task_count + retired;
        --clamp->counts[position];
        if (clamp->counts[position] == 0U) {
            const uint32_t device = problem->alias_device[reload->alias];
            clamp->used[(uint64_t)device * facts->boundary_count + retired + 1U] -=
                problem->alias_size_bytes[reload->alias];
        }
    }
}

/* Delay fetches at one boundary until it fits. Running out of fetches to
 * delay is not an error: the boundary is as relieved as moving fetches can
 * make it, and what remains is the reducer's problem rather than this one. */
static void relieve_boundary(
    const ShadowSpillScheduleFacts *facts,
    const uint8_t *resident,
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
        const uint32_t selected = shadowspill_schedule_first_active_reload(
            clamp->active, clamp->word_count, boundary, clamp->ranked
        );
        if (selected == UINT32_MAX) {
            return;
        }
        delay_reload(
            facts,
            resident,
            &reloads[selected],
            clamp->rank_by_reload[selected],
            boundary,
            clamp
        );
    }
}

int shadowspill_schedule_clamp_triggers_to_fit(
    const ShadowSpillScheduleFacts *facts,
    const uint8_t *resident,
    const uint8_t *breaks,
    Reload *reloads,
    uint32_t reload_count,
    int fetch_headroom
) {
    Clamp clamp;
    if (clamp_create(
            facts, resident, breaks, reload_count, fetch_headroom, &clamp
        ) != 0) {
        clamp_destroy(&clamp);
        return -1;
    }
    charge_fetch_windows(facts, resident, reloads, reload_count, &clamp);
    rank_reloads(reload_count, &clamp);
    mark_movable_reloads(facts, resident, reloads, reload_count, &clamp);
    for (uint32_t device = 0U; device < facts->device_count; ++device) {
        for (uint32_t boundary = 0U; boundary < facts->task_count; ++boundary) {
            relieve_boundary(facts, resident, reloads, device, boundary, &clamp);
        }
    }
    clamp_destroy(&clamp);
    return 0;
}
